#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
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

TVector2 computeMETvec(const std::vector<TLorentzVector>& visible){
  TVector2 met(0,0);
  for(const auto& v : visible){
    met -= TVector2(v.Px(), v.Py());
  }
  return met;
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

  double Ctau1 = tau1.M2() - 1.77686*1.77686; // tau mass^2 in GeV^2
  double Ctau2 = tau2.M2() - 1.77686*1.77686;

  double CB0 = Bfit.M2() - 5.2797*5.2797; // B0 mass^2 in GeV^2

  // Build chi2
  double chi2 = (Cx*Cx + Cy*Cy)/(in.sigmaMET*in.sigmaMET)
              + (Ctau1*Ctau1 + Ctau2*Ctau2)/(in.sigmaTau*in.sigmaTau)
              + (Cpoint*Cpoint)/(in.sigmaPoint*in.sigmaPoint)
              + (CB0*CB0)/(in.sigmaB0*in.sigmaB0);
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
  // Number of events
  int nEvents = 3000;
  if (argc > 1) nEvents = atoi(argv[1]);

  // 1) Configure Pythia
  Pythia pythia;
  pythia.readString("Beams:idA = 2212");
  pythia.readString("Beams:idB = 2212");
  pythia.readString("Beams:eCM = 13000.");
  pythia.readString("Random:seed = 22");           // set seed
  pythia.readString("HardQCD:gg2bbbar = on");
  pythia.readString("HardQCD:qqbar2bbbar = on");
  pythia.readString("PhaseSpace:pTHatMin = 5.");
  pythia.readString("Beams:allowVertexSpread = on");
  pythia.readString("Beams:sigmaVertexX = 0.01");
  pythia.readString("Beams:sigmaVertexY = 0.01");
  pythia.readString("Beams:sigmaVertexZ = 50.0");
  pythia.readString("ParticleDecays:limitTau0 = on");
  pythia.readString("511:mayDecay = off");
  pythia.readString("-511:mayDecay = off");
  pythia.init();

  // 2) Initialize EvtGen
  const char* decayFile = "/home/pablo/projects/bachelor_thesis_cpp/Specific_background.dec";
  const char* pdlFile   = "/home/pablo/evtgen-install/share/EvtGen/evt.pdl";
  EvtExternalGenList genList(true, "/home/pablo/pythia8-install/share/Pythia8/xmldoc", "", true);
  EvtAbsRadCorr* photos = genList.getPhotosModel();
  std::list<EvtDecayBase*> externals = genList.getListOfModels();
  EvtRandomEngine* randEng = new EvtSimpleRandomEngine();
  EvtGen* evtgen = new EvtGen(decayFile, pdlFile, randEng, photos, &externals, 1, false);

  // 3) Output file and tree
  TFile outFile("Specific_Background.root", "RECREATE");
  TTree tree("Events", "Simulation of Specific Backgrounds");
  // Branch definitions
  Float_t ptB, etaB, phiB;
  Float_t PVx, PVy, PVz, PVxErr=0.01f, PVyErr=0.01f, PVzErr=0.025f;
  Float_t SVx, SVy, SVz, SVxErr, SVyErr, SVzErr, vertexChi2;
  Float_t kst_pt, kst_eta, kst_phi;
  Float_t tauPlus_pt, tauPlus_eta, tauPlus_phi;
  Float_t tauMinus_pt, tauMinus_eta, tauMinus_phi;
  Float_t m_tauPlus, m_tauMinus, m_kst;
  tree.Branch("ptB", &ptB, "ptB/F");
  tree.Branch("etaB", &etaB, "etaB/F");
  tree.Branch("phiB", &phiB, "phiB/F");
  tree.Branch("PVx", &PVx, "PVx/F");
  tree.Branch("PVy", &PVy, "PVy/F");
  tree.Branch("PVz", &PVz, "PVz/F");
  tree.Branch("PVxErr", &PVxErr, "PVxErr/F");
  tree.Branch("PVyErr", &PVyErr, "PVyErr/F");
  tree.Branch("PVzErr", &PVzErr, "PVzErr/F");
  tree.Branch("SVx", &SVx, "SVx/F");
  tree.Branch("SVy", &SVy, "SVy/F");
  tree.Branch("SVz", &SVz, "SVz/F");
  tree.Branch("SVxErr", &SVxErr, "SVxErr/F");
  tree.Branch("SVyErr", &SVyErr, "SVyErr/F");
  tree.Branch("SVzErr", &SVzErr, "SVzErr/F");
  tree.Branch("vertexChi2", &vertexChi2, "vertexChi2/F");
  tree.Branch("kst_pt", &kst_pt, "kst_pt/F");
  tree.Branch("kst_eta", &kst_eta, "kst_eta/F");
  tree.Branch("kst_phi", &kst_phi, "kst_phi/F");
  tree.Branch("tauPlus_pt", &tauPlus_pt, "tauPlus_pt/F");
  tree.Branch("tauPlus_eta", &tauPlus_eta, "tauPlus_eta/F");
  tree.Branch("tauPlus_phi", &tauPlus_phi, "tauPlus_phi/F");
  tree.Branch("tauMinus_pt", &tauMinus_pt, "tauMinus_pt/F");
  tree.Branch("tauMinus_eta", &tauMinus_eta, "tauMinus_eta/F");
  tree.Branch("tauMinus_phi", &tauMinus_phi, "tauMinus_phi/F");
  tree.Branch("m_tauPlus", &m_tauPlus, "m_tauPlus/F");
  tree.Branch("m_tauMinus", &m_tauMinus, "m_tauMinus/F");
  tree.Branch("m_kst", &m_kst, "m_kst/F");

  std::mt19937 rng(41);
  std::normal_distribution<double> smearSVxy(0.01);
  std::normal_distribution<double> smearSVz(0.01);
  std::chi_squared_distribution<double> chi2Dist(1);

  // 4) Event loop
  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    if (!pythia.next()) continue;
    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if ((p.id()==511 || p.id()==-511) && p.isFinal()) {
        // Reset
        kst_pt = kst_eta = kst_phi = -999.f;
        tauPlus_pt = tauPlus_eta = tauPlus_phi = -999.f;
        tauMinus_pt = tauMinus_eta = tauMinus_phi = -999.f;

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
        double chi2ndf = chi2/1;  // Since DOF = 6
        vertexChi2 = chi2ndf;

        // EvtGen decay
        EvtVector4R  mom( p.e(), p.px(), p.py(), p.pz() );
        EvtParticle* evtB = EvtParticleFactory::particleFactory(
                      EvtPDL::evtIdFromStdHep(p.id()), mom );
        evtgen->generateDecay(evtB);

        bool gotKstar = false, gotTauplus = false, gotTauminus = false;

        // ─── 5.d) Walk the decay tree to pull out your final‑state taus, K*, etc.
        std::vector<EvtParticle*> finalDau;
        std::function<void(EvtParticle*)> collect = [&](EvtParticle* par){
          if (par->getNDaug()==0) {
            finalDau.push_back(par);
          } else {
            for (int d=0; d<par->getNDaug(); ++d)
              collect(par->getDaug(d));
          }
        };
        collect(evtB);


        //NOTE: Maybe use recursive search for particles instead of leaf search to get all the particles!!
        // Fill daughter kinematics
        for (auto dau : finalDau) {
          EvtId id = dau->getId();
          auto p4 = dau->getP4Lab();
          TLorentzVector v(p4.get(1), p4.get(2), p4.get(3), p4.get(0));
          if (id == EvtPDL::getId("K*0") || id == EvtPDL::getId("anti-K*0")) {
            kst_pt = v.Pt(); kst_eta = v.Eta(); kst_phi = v.Phi();
            gotKstar = true;
          } else if (id == EvtPDL::getId("tau+")) {
            tauPlus_pt = v.Pt(); tauPlus_eta = v.Eta(); tauPlus_phi = v.Phi();
            gotTauplus = true;
          } else if (id == EvtPDL::getId("mu-")) {
            tauMinus_pt = v.Pt(); tauMinus_eta = v.Eta(); tauMinus_phi = v.Phi();
            gotTauminus = true;
          }
        }
        if (gotKstar && gotTauplus && gotTauminus) {
          tree.Fill();
          }
        delete evtB;
      }
    }
  }

  std::cout << "Entries: " << tree.GetEntries() << std::endl;
  pythia.stat();
  outFile.Write();
  outFile.Close();
  return 0;
}
