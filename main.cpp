// main.cc
// Example Pythia8 driver to generate stable B0 mesons for later EvtGen decay
// Includes primary (PV) and secondary (SV) vertex computation with smearing

// CMakeLists.txt snippet:
// ------------------------------------------------------
// cmake_minimum_required(VERSION 3.15)
// project(B0Generator)
// find_package(Pythia8 REQUIRED)
// find_package(ROOT REQUIRED COMPONENTS RIO Tree)
// add_executable(runPythia main.cc)
// target_include_directories(runPythia PRIVATE ${PYTHIA8_INCLUDE_DIRS} ${ROOT_INCLUDE_DIRS})
// target_link_libraries(runPythia PRIVATE ${PYTHIA8_LIBRARIES} ${ROOT_LIBRARIES})
// ------------------------------------------------------

#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include <utility>
#include <random>
#include <iostream>
#include <EvtGen/EvtGen.hh>
#include <EvtGenBase/EvtParticleFactory.hh>
#include <EvtGenBase/EvtPDL.hh>
#include <EvtGenBase/EvtRandomEngine.hh>
#include <EvtGenBase/EvtSimpleRandomEngine.hh>
#include <EvtGenExternal/EvtPythiaEngine.hh>
#include "EvtGenExternal/EvtExternalGenList.hh"
#include "EvtGenBase/EvtAbsRadCorr.hh"
#include "EvtGenBase/EvtDecayBase.hh"

inline TVector2 computeMETvec(const std::vector<TLorentzVector>& visible){
  TVector2 met(0,0);
  for(const auto& v : visible){
    met -= TVector2(v.Px(), v.Py());
  }
  return met;
}

inline double transverseMass(const TLorentzVector& visSum, const TVector2& metVec) {
  double m_vis = visSum.M();
  double pT_vis = visSum.Pt();
  double E_T_vis = std::sqrt(m_vis*m_vis + pT_vis*pT_vis);
  double ET_miss = metVec.Mod();
  double dot = metVec.X()*visSum.Px() + metVec.Y()*visSum.Py();
  double mT2 = m_vis*m_vis + 2.0 * (E_T_vis * ET_miss - dot);
  return (mT2 > 0 ? std::sqrt(mT2) : 0.0); // ? operator: (condition ? ifTrue : ifFalse)
}

// function to do double collinear approximation reconstruction on both tau branches at the same time
inline std::tuple<bool, TLorentzVector, TLorentzVector> reconstructTauDoubleCollinear(const TLorentzVector& visSum1, const TLorentzVector& visSum2, const TVector2& metVec) {
  const double pT_vis1 = visSum1.Pt();
  const double pT_vis2 = visSum2.Pt();
  if (pT_vis1 <= 0 || pT_vis2 <= 0) return {false, visSum1, visSum2};  // no visible pT → nothing to do

  // unit vector along the visible momentum in the transverse plane
  TVector2 u1(visSum1.Px(), visSum1.Py());
  TVector2 u2(visSum2.Px(), visSum2.Py());
  double det = u1.X()*u2.Y() - u1.Y()*u2.X();
  /*if (std::abs(det)<1e-6) {
    return {false, visSum1, visSum2};
  }*/

  // project MET onto that unit vector
  double a1 = ( metVec.X()*u2.Y() - metVec.Y()*u2.X() )/det;
  double a2 = (-metVec.X()*u1.Y() + metVec.Y()*u1.X() )/det;
  double x1 = 1/(1 + a1);
  double x2 = 1/(1 + a2);
  if (x1 <= 0 || x2 <= 0) {
    return {false, visSum1, visSum2};
  }
  else if (x1 > 1 || x2 > 1) {return {false, visSum1, visSum2};}

  // build a massless neutrino 4-vector (pz = 0)
  TLorentzVector nu1;
  nu1.SetPxPyPzE(u1.X()*x1, u1.Y()*x1, 0.0, a1);
  TLorentzVector nu2;
  nu2.SetPxPyPzE(u2.X()*x2, u2.Y()*x2, 0.0, a2);

  // tau = visible + neutrino
  return {true, visSum1 + nu1, visSum2 + nu2};
}

// do function for collinear approximation reconstruction for a single tau branch
inline std::pair<bool, TLorentzVector> reconstructTauCollinear( const TLorentzVector& visSum, TVector2& metVec) {
  const double pT_vis = visSum.Pt();
  if (pT_vis <= 0) return {false, visSum};  // no visible pT → nothing to do

  TVector2 u(visSum.Px()/pT_vis, visSum.Py()/pT_vis); // unit vector
  const double alpha = metVec.X()*u.X() + metVec.Y()*u.Y(); // project MET onto that unit vector
  if (alpha <= 0.0) {
    return {false, visSum};
  }
  TLorentzVector nu;
  nu.SetPxPyPzE(u.X()*alpha, u.Y()*alpha, 0.0, alpha); // construct neutrino track
  return {true, visSum + nu};
}

// do function do try tau reconstruction but with transverse mass (if too low pt for collinear approx)
inline TLorentzVector reconstructTauTransverseMass( const TLorentzVector& visSum, const TVector2& metVec) {
  double mT = transverseMass(visSum, metVec);
  double px_tau = visSum.Px() + metVec.X();
  double py_tau = visSum.Py() + metVec.Y();
  double pz_tau = visSum.Pz();
  double p2_tau = px_tau*px_tau + py_tau*py_tau + pz_tau*pz_tau;
  double E_tau  = std::sqrt(p2_tau + mT*mT);
  TLorentzVector tau;
  tau.SetPxPyPzE(px_tau, py_tau, pz_tau, E_tau);
  return tau;
}

struct FitInputs {
  TLorentzVector vis1, vis2, kst;
  double METx, METy;
  double sigmaMET, sigmaTau, sigmaPoint, sigmaB0;
  TVector3 PV, SV;
};

double fitfunction(const double *par, double *grad, void *fdata) {
  //    par[0..2] = (p_nu1_x, p_nu1_y, p_nu1_z)
  //    par[3..5] = (p_nu2_x, p_nu2_y, p_nu2_z)
  auto const &in = *static_cast<FitInputs*>(fdata);
  TLorentzVector nu1{ par[0], par[1], par[2], // build neutrino vectors
    std::hypot(par[0], par[1], par[2]) };
  TLorentzVector nu2{ par[3], par[4], par[5],
    std::hypot(par[3], par[4], par[5]) };

  double E_vis1 = in.vis1.E();        // lab-frame energy of visible system
  double p_vis1 = in.vis1.P();        // lab-frame |p| of visible system
  double m_vis1 = in.vis1.M();        // invariant mass of visible system
  double E_vis2 = in.vis2.E();        // lab-frame energy of visible system
  double p_vis2 = in.vis2.P();        // lab-frame |p| of visible system
  double m_vis2 = in.vis2.M();        // invariant mass of visible system
  double mTau = 1.77686;  //GeV
  double Evis1Star = (mTau*mTau + m_vis1*m_vis1) / (2.0 * mTau);
  double pvis1Star = (mTau*mTau - m_vis1*m_vis1) / (2.0 * mTau);
  double Evis2Star = (mTau*mTau + m_vis2*m_vis2) / (2.0 * mTau);
  double pvis2Star = (mTau*mTau - m_vis2*m_vis2) / (2.0 * mTau);
  double R_lab1 = p_vis1 / E_vis1;
  double R_lab2 = p_vis2 / E_vis2;
  double numerator1   = pvis1Star - R_lab1 * Evis1Star;
  double denominator1 = R_lab1 * pvis1Star - Evis1Star;
  double numerator2   = pvis2Star - R_lab2 * Evis2Star;
  double denominator2 = R_lab2 * pvis2Star - Evis2Star;
  double beta1 = numerator1 / denominator1;
  double beta2 = numerator2 / denominator2;
  double gamma1 = 1.0 / TMath::Sqrt(1.0 - beta1 * beta1);
  double gamma2 = 1.0 / TMath::Sqrt(1.0 - beta2 * beta2);
  double pInv1Star = pvis1Star;  // = (m_tau^2 - m_vis^2)/(2 m_tau)
  double pInv2Star = pvis2Star;
  double pNu_lab1 = gamma1 * pInv1Star * (1.0 + beta1);
  double pNu_lab2 = gamma2 * pInv2Star * (1.0 + beta2);

  double CpNu1 = pNu_lab1 - std::hypot(par[0], par[1], par[2]);
  double CpNu2 = pNu_lab2 - std::hypot(par[3], par[4], par[5]);
  double sigmapNu = 0.1; // in GeV

  //MET constraints
  double Cx = (par[0]+par[3] - in.METx);
  double Cy = (par[1]+par[4] - in.METy);

  //Tau mass constraints
  auto tau1 = in.vis1 + nu1;
  auto tau2 = in.vis2 + nu2;
  TLorentzVector Bfit = in.kst + tau1 + tau2;

  // flight direction of B meson, original and reconstructed
  TVector3 flight = in.SV - in.PV;
  TVector3 u      = flight.Unit();
  TVector3 pB = TVector3(Bfit.Px(), Bfit.Py(), Bfit.Pz());

  double Cpoint = pB.Cross(u).Mag(); // |pB x u| minimal if collinear, what we want

  double Ctau1 = 1.77686 - tau1.M(); // tau mass in GeV
  double Ctau2 = 1.77686 - tau2.M();

  // Penalize large pz neutrino components
  double sigmaPz = 10.0; // in GeV
  double Cz1 = par[2];
  double Cz2 = par[5];

  double CB0 = Bfit.M2() - 5.2797*5.2797; // B0 mass^2 in GeV^2

  // Build chi2
  double chi2 = (Cx*Cx + Cy*Cy)/(in.sigmaMET*in.sigmaMET)
              + (Ctau1*Ctau1 + Ctau2*Ctau2)/(in.sigmaTau*in.sigmaTau)
              /*+ (Cz1*Cz1 + Cz2*Cz2)/(sigmaPz*sigmaPz)*/
              + (Cpoint*Cpoint)/(in.sigmaPoint*in.sigmaPoint)
              + (CpNu1*CpNu1 + CpNu2*CpNu2)/(sigmapNu);
              // + (CB0*CB0)/(in.sigmaB0*in.sigmaB0);
  return chi2;
}

std::array<TLorentzVector,2> FitNeutrinos(const FitInputs &in) {
  // Create Minuit2 minimizer
  auto minim = ROOT::Math::Factory::CreateMinimizer("Minuit2","Migrad");

  // Wrap old fitfunction into a lambda of the right shape
  auto wrappedFCN = [&](double const* par) -> double {
    return fitfunction(par, nullptr, const_cast<FitInputs*>(&in));
  };
  // Wrap fitfuntion
  ROOT::Math::Functor functor(wrappedFCN, /*ndim=*/6);
  minim->SetFunction(functor);

  // Initial guesses
  double step = 0.1;
  minim->SetVariable(0,"p1_x", in.METx/2, step);
  minim->SetVariable(1,"p1_y", in.METy/2, step);
  minim->SetVariable(2,"p1_z", 0, step);
  minim->SetVariable(3,"p2_x", in.METx/2, step);
  minim->SetVariable(4,"p2_y", in.METy/2, step);
  minim->SetVariable(5,"p2_z", 0, step);

  // Run MIGRAD
  minim->Minimize();

  // Retrieve results
  const double *res = minim->X();
  TLorentzVector nu1{ res[0], res[1], res[2],
    std::hypot(res[0],res[1],res[2]) };
  TLorentzVector nu2{ res[3], res[4], res[5],
    std::hypot(res[3],res[4],res[5]) };

  return { nu1, nu2 };
}

using namespace Pythia8;

int main(int argc, char* argv[]) {
  // Number of events (default 100k if not passed)
  int nEvents = 5000;
  if (argc > 1) nEvents = atoi(argv[1]);

  // Configure Pythia for pp collisions @ 13 TeV, b-quark production
  Pythia pythia;

  pythia.readString("Beams:idA = 2212");           // proton
  pythia.readString("Beams:idB = 2212");           // proton
  pythia.readString("Beams:eCM = 13000.");         // 13 TeV
  pythia.readString("Random:seed = 22");           // set seed
  pythia.readString("HardQCD:gg2bbbar = on");      // turn on gg->bb
  pythia.readString("HardQCD:qqbar2bbbar = on");   // turn on qqbar->bb
  pythia.readString("PhaseSpace:pTHatMin = 5.");   // pT hat cut (GeV)

  // Vertex smearing for primary vertex (Gaussian)
  pythia.readString("Beams:allowVertexSpread = on");
  pythia.readString("Beams:sigmaVertexX = 0.01");         // mm (PV resolution)
  pythia.readString("Beams:sigmaVertexY = 0.01");         // mm
  pythia.readString("Beams:sigmaVertexZ = 0.025");         // mm

  // Keep B0 mesons stable (will decay later via EvtGen)
  pythia.readString("ParticleDecays:limitTau0 = on"); // limit very short lifetimes
  pythia.readString("511:mayDecay = off");          // B0 (PDG 511) stable
  pythia.readString("-511:mayDecay = off");         // anti-B0

  // Initialize pythia
  pythia.init();

  // ─── Initialize EvtGen ──────────────────────────────────────────────
  const char* decayFile = "/home/pablo/projects/bachelor_thesis_cpp/B0_Kst_tautau.dec";
  const char* pdlFile   = "/home/pablo/evtgen-install/share/EvtGen/evt.pdl";

  // Decide whether to translate Pythia codes, where your xmldoc lives, etc.
  bool convertPythiaCodes = true;
  std::string  pythiaXmlDir   = "/home/pablo/pythia8-install/share/Pythia8/xmldoc";
  bool useEvtGenRnd = true;

  // Build the standard external list:
  EvtExternalGenList genList(convertPythiaCodes,
                             pythiaXmlDir,
                             /*photonType=*/"",
                             useEvtGenRnd);
  // Grab the Photos radiative‐correction engine and the other external models
  EvtAbsRadCorr*        photos = genList.getPhotosModel();
  std::list<EvtDecayBase*> externals = genList.getListOfModels();

  // random engine for EvtGen
  EvtRandomEngine* randEng = new EvtSimpleRandomEngine();
  // Now construct EvtGen with *your* decay files, PDG table, RNG **and** externals
  EvtGen* evtgen = new EvtGen(
    std::string(decayFile),
    std::string(pdlFile),
    randEng, photos, &externals, 1, false
  );


  // 4) Prepare output ROOT file & TTree for B0 kinematics + vertices + uncertainties
  TFile outFile("Simulation_Data_Smeared.root", "RECREATE");
  TTree tree("Events", "B0 -> Kst Tau Tau production with vertices and uncertainties");

  // Kinematics
  Float_t ptB, etaB, phiB;
  // Primary vertex (PV) from Pythia, smeared
  Float_t PVx, PVy, PVz;
  // PV uncertainties (mm)
  Float_t PVxErr, PVyErr, PVzErr;
  // Secondary vertex (SV) computed via lifetime propagation + measurement smearing
  Float_t SVx, SVy, SVz;
  // SV uncertainties (mm)
  Float_t SVxErr, SVyErr, SVzErr;
  // Vertex fit quality (chi2)
  Float_t vertexChi2;
  // PV uncertainties from vertex smearing parameters
  PVxErr = 0.01;  // mm (sigmaX)
  PVyErr = 0.01;  // mm (sigmaY)
  PVzErr = 0.025;  // mm (sigmaZ)

  tree.Branch("ptB",       &ptB,       "ptB/F");
  tree.Branch("etaB",      &etaB,      "etaB/F");
  tree.Branch("phiB",      &phiB,      "phiB/F");
  tree.Branch("PVx",       &PVx,       "PVx/F");
  tree.Branch("PVy",       &PVy,       "PVy/F");
  tree.Branch("PVz",       &PVz,       "PVz/F");
  tree.Branch("PVxErr",    &PVxErr,    "PVxErr/F");
  tree.Branch("PVyErr",    &PVyErr,    "PVyErr/F");
  tree.Branch("PVzErr",    &PVzErr,    "PVzErr/F");
  tree.Branch("SVx",       &SVx,       "SVx/F");
  tree.Branch("SVy",       &SVy,       "SVy/F");
  tree.Branch("SVz",       &SVz,       "SVz/F");
  tree.Branch("SVxErr",    &SVxErr,    "SVxErr/F");
  tree.Branch("SVyErr",    &SVyErr,    "SVyErr/F");
  tree.Branch("SVzErr",    &SVzErr,    "SVzErr/F");
  tree.Branch("vertexChi2", &vertexChi2, "vertexChi2/F");

  // Single-entry daughter kinematics
  Float_t kst_pt, kst_eta, kst_phi;
  Float_t tauPlus_pt, tauPlus_eta, tauPlus_phi;
  Float_t tauMinus_pt, tauMinus_eta, tauMinus_phi;
  Float_t m_tauMinus, m_tauPlus, m_kst;
  Float_t mT_tautau, m_tautau_coll;


  tree.Branch("kst_pt", &kst_pt, "kst_pt/F");
  tree.Branch("kst_eta", &kst_eta, "kst_eta/F");
  tree.Branch("kst_phi", &kst_phi, "kst_phi/F");
  tree.Branch("tauPlus_pt", &tauPlus_pt, "tauPlus_pt/F");
  tree.Branch("tauPlus_eta", &tauPlus_eta, "tauPlus_eta/F");
  tree.Branch("tauPlus_phi", &tauPlus_phi, "tauPlus_phi/F");
  tree.Branch("tauMinus_pt", &tauMinus_pt, "tauMinus_pt/F");
  tree.Branch("tauMinus_eta", &tauMinus_eta, "tauMinus_eta/F");
  tree.Branch("tauMinus_phi", &tauMinus_phi, "tauMinus_phi/F");
  tree.Branch("m_tauMinus", &m_tauMinus, "m_tauMinus/F");
  tree.Branch("m_tauPlus", &m_tauPlus, "m_tauPlus/F");
  tree.Branch("m_kst", &m_kst, "m_kst/F");
  tree.Branch("mT_tautau", &mT_tautau, "mT_tautau/F");
  tree.Branch("m_tautau_coll", &m_tautau_coll, "m_tautau_coll/F");

  // Random number generators for measurement smearing and uncertainties
  std::mt19937 rng(42);
  std::normal_distribution<double> smearSVxy(0.0, 0.01);    // mm (SV spatial resolution)
  std::normal_distribution<double> smearSVz(0.0, 0.01);      // mm
  std::chi_squared_distribution<double> chi2Dist(1);   // chi2 with n DOF



  // Event loop: find all B0 in the event record
  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    if (!pythia.next()) continue;

    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if ((p.id() == 511 || p.id() == -511) && p.isFinal()) {
        // Initialize daughter variables to a default invalid value
        kst_pt = kst_eta = kst_phi = -999.;
        tauPlus_pt = tauPlus_eta = tauPlus_phi = -999.;
        tauMinus_pt = tauMinus_eta = tauMinus_phi = -999.;

        // B0 Kinematics
        ptB  = p.pT();
        etaB = p.eta();
        phiB = p.phi();
        // B0 Primary vertex (mm)
        PVx = p.xProd();
        PVy = p.yProd();
        PVz = p.zProd();
        // Compute B0 secondary vertex via decay length
        double properCTau  = p.tau();
        double betaGamma   = p.pAbs() / p.m();
        double decayLength = properCTau * betaGamma;

        // Get the B0’s momentum components
        double px   = p.px();
        double py   = p.py();
        double pz   = p.pz();

        // Compute the magnitude of the B0 3‑momentum
        double pMag = std::sqrt(px*px + py*py + pz*pz);

        // Build the unit direction vector of B0
        double dirX = px / pMag;
        double dirY = py / pMag;
        double dirZ = pz / pMag;

        double SVx_true = PVx + dirX * decayLength;
        double SVy_true = PVy + dirY * decayLength;
        double SVz_true = PVz + dirZ * decayLength;

        // 5.d) Measurement smearing of SV
        SVx = SVx_true + smearSVxy(rng);
        SVy = SVy_true + smearSVxy(rng);
        SVz = SVz_true + smearSVz(rng);
        // 5.e) Assign uncertainties and chi2
        SVxErr = smearSVxy.stddev();
        SVyErr = smearSVxy.stddev();
        SVzErr = smearSVz.stddev();
        double chi2 = chi2Dist(rng);
        double chi2ndf = chi2 / 1.0;  // Since DOF = n
        vertexChi2 = chi2ndf;

        // ─── Hand this B0 to EvtGen ────────────────────────────
        //   Create an EvtGen “particle” from the Pythia entry:
        EvtVector4R  mom( p.e(), p.px(), p.py(), p.pz() );
        EvtParticle* evtB = EvtParticleFactory::particleFactory(
                      EvtPDL::evtIdFromStdHep(p.id()), mom );
        evtgen->generateDecay(evtB);

        // ─── find the *intermediate* daughters of the B ───────────────
        EvtParticle* kstar = nullptr;
        EvtParticle* tauP  = nullptr;
        EvtParticle* tauM  = nullptr;
        for(int i=0; i<evtB->getNDaug(); ++i){
          EvtParticle* d = evtB->getDaug(i);
          EvtId id = d->getId();
          if      (id == EvtPDL::getId("K*0") || id == EvtPDL::getId("anti-K*0") ) kstar = d;
          else if (id == EvtPDL::getId("tau+" )    ) tauP  = d;
          else if (id == EvtPDL::getId("tau-" )    ) tauM  = d;
        }

        std::cout << "[DEBUG] Found branches:"
          << " K*0="<<(kstar? "yes":"NO")
          << " tau+="<<(tauP? "yes":"NO")
          << " tau-="<<(tauM? "yes":"NO")<<"\n";
        if (!kstar || !tauP || !tauM) {
          std::cerr << "[WARN] incomplete B→K*ττ decay, skipping this event\n";
          delete evtB;
          continue;
        }


        // ─── recurse to collect _all_ final‐state leaves under a node ──
        auto collectLeaves = [&](EvtParticle* node,
                                 std::vector<EvtParticle*>& out){
          std::function<void(EvtParticle*)> go = [&](EvtParticle* p){
            if (p->getNDaug()==0) {
              out.push_back(p);
            } else {
              for(int i=0;i<p->getNDaug();++i)
                go(p->getDaug(i));
            }
          };
          go(node);
        };

        // ─── pull out your leaves for each branch ───────────────────────
        std::vector<EvtParticle*> kstarLeaves, tauPLeaves, tauMLeaves;
        collectLeaves(kstar, kstarLeaves);
        collectLeaves(tauP,  tauPLeaves);
        collectLeaves(tauM,  tauMLeaves);

        // classify each tau branch
        auto classifyTau = [&](const std::vector<EvtParticle*>& leaves){
          int nMuon = 0, nPion = 0;
          for(auto p : leaves){
            int pdg = std::abs(EvtPDL::getStdHep(p->getId()));
            if      (pdg == 13)      ++nMuon;
            else if (pdg == 211)     ++nPion;
          }
          // exactly one muon => muonic decay
          if (nMuon == 1 && leaves.size() >= 2) return "MU";
          // exactly three charged pions => 3π decay
          if (nPion == 3 && leaves.size() == 4) return "3P";
          return "OTHER";
        };

        std::string typeP = classifyTau(tauPLeaves);
        std::string typeM = classifyTau(tauMLeaves);

        std::cout << "[DEBUG] τ+ decay = " << typeP
                  << ", τ- decay = " << typeM << "\n";

        // require exactly one MU and one 3P
        if (!((typeP=="MU" && typeM=="3P") ||
              (typeP=="3P"&& typeM=="MU")) ){
          std::cout<<"[INFO] skipping event: not μ+3π\n";
          delete evtB;
          continue;
              }

        // collect all visible tracks from both tau decays (3prong, muon)
        std::vector<TLorentzVector> tauVis;
        tauVis.reserve(6);

        auto addToTauDaughters = [&](const std::vector<EvtParticle*>& leaves){
          for (auto p : leaves) {
            int pdg = std::abs(EvtPDL::getStdHep(p->getId()));
            if (pdg==12||pdg==14||pdg==16) continue;
            auto p4 = p->getP4Lab();
            tauVis.emplace_back(p4.get(1),p4.get(2),p4.get(3),p4.get(0));
          }
        };
        addToTauDaughters(tauPLeaves);
        addToTauDaughters(tauMLeaves);

        // compute missing transverse energy, globally
        TVector2 metGlobal = computeMETvec(tauVis);


        // define a little helper to sum and collect only non‑neutrino four‑vectors:
        auto sumVisible = [&](const std::vector<EvtParticle*>& leaves) -> std::pair<TLorentzVector, std::vector<TLorentzVector>>{
          TLorentzVector sum(0,0,0,0);
          std::vector<TLorentzVector> visTracks;
          for(auto dau : leaves){
            int pdg = EvtPDL::getStdHep(dau->getId());
            std::cout << "[DEBUG]   leaf PDG="<<pdg<<"\n";
            if (std::abs(pdg)==12 || std::abs(pdg)==14 || std::abs(pdg)==16) continue;
            auto p4 = dau->getP4Lab();
            TLorentzVector v(p4.get(1), p4.get(2), p4.get(3), p4.get(0));
            sum += v;
            visTracks.push_back(v);
          }
          return std::make_pair(sum, visTracks);
        };

        // relative pt error for low pt << 100GeV
        double sigma_pt_rel = 0.007;

        // build your reco‑K* vector and fill:
        TLorentzVector kstarReco = sumVisible(kstarLeaves).first;
        double sigma_pt_Kst = kstarReco.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_Kst(kstarReco.Pt(), sigma_pt_Kst);
        kst_pt  = smearPt_Kst(rng);
        kst_eta = kstarReco.Eta();
        kst_phi = kstarReco.Phi();
        m_kst = kstarReco.M();

        // same for the tau branches
        TLorentzVector tauMvisSum = sumVisible(tauMLeaves).first;
        TLorentzVector tauPvisSum = sumVisible(tauPLeaves).first;

        // compute transverse mass of ditau system and fill
        mT_tautau = transverseMass(tauMvisSum + tauPvisSum, metGlobal);

        //build input structure called FitInputs
        FitInputs in;
        in.vis1 = tauMvisSum;
        in.vis2 = tauPvisSum;
        in.kst = kstarReco;
        in.METx = metGlobal.X();
        in.METy = metGlobal.Y();
        in.sigmaMET = 0.1; // in GeV
        in.sigmaTau = 0.1; // in GeV
        in.sigmaB0 = 0.3; // in GeV
        in.sigmaPoint = 0.1; // in GeV
        in.PV = TVector3 (PVx, PVy, PVz);
        in.SV = TVector3 (SVx, SVy, SVz);

        auto neutrinos = FitNeutrinos(in);
        auto &nu1 = neutrinos[0];
        auto &nu2 = neutrinos[1];

        TLorentzVector tauMvis = in.vis1 + nu1;
        TLorentzVector tauPvis = in.vis2 + nu2;

        double sigma_pt_tauP = tauPvisSum.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_tauP(tauPvisSum.Pt(), sigma_pt_tauP);
        tauPlus_pt  = smearPt_tauP(rng);
        tauPlus_eta = tauPvisSum.Eta();
        tauPlus_phi = tauPvisSum.Phi();
        m_tauPlus = tauPvisSum.M();
        std::cout << "[DEBUG] mass of Tau Plus: " << m_tauPlus << "\n";

        // …and for the τ‑ branch:
        double sigma_pt_tauM = tauMvisSum.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_tauM(tauMvisSum.Pt(), sigma_pt_tauM);
        tauMinus_pt  = smearPt_tauM(rng);
        tauMinus_eta = tauMvisSum.Eta();
        tauMinus_phi = tauMvisSum.Phi();
        m_tauMinus = tauMvisSum.M();
        std::cout << "[DEBUG] mass of Tau Minus: " << m_tauMinus << "\n";


        tree.Fill();

        // destroy the EvtGen particle to free memory:
        delete evtB;
      }
    }
  }

  std::cout << "Entries in tree: " << tree.GetEntries() << "\n";

  // Finalize
  pythia.stat();      // print summary to stdout
  outFile.Write();    // save TTree
  outFile.Close();

  return 0;
}
