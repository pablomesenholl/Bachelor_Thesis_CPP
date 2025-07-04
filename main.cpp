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

struct Track {
  TVector3 origin;
  TVector3 direction;
};

// Chi2 function: sum of squared perpendicular distances to each track
class VertexChi2Functor {
public:
  VertexChi2Functor(const std::vector<Track>& tracks, double sigma)
    : fTracks(tracks), fSigma2(sigma * sigma) {}

  double operator()(const double* x) const {
    TVector3 V(x[0], x[1], x[2]);
    double chi2 = 0.0;
    for (const auto& t : fTracks) {
      TVector3 diff = V - t.origin;
      TVector3 perp = diff - diff.Dot(t.direction) * t.direction;
      chi2 += perp.Mag2() / fSigma2;
    }
    return chi2;
  }

private:
  const std::vector<Track>& fTracks;
  double fSigma2;
};

// Function to perform simplified vertex fit
bool FitVertex(const std::vector<Track>& tracks, double sigma, TVector3& fittedVertex, Float_t& chi2Out) {
  if (tracks.size() < 2) return false;

  // --- build a “composite” seed from the track origins, average starting point of tracks ---
  TVector3 seed(0,0,0);
  for (auto& t : tracks) seed += t.origin;
  seed *= (1.0 / tracks.size());

  ROOT::Math::Functor functor(VertexChi2Functor(tracks, sigma), 3);
  auto minimizer = std::unique_ptr<ROOT::Math::Minimizer>(
    ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));

  minimizer->SetFunction(functor);
  minimizer->SetMaxFunctionCalls(200);
  minimizer->SetMaxIterations(500);
  minimizer->SetTolerance(1e-4);

  // Set initial guess (e.g. origin)
  minimizer->SetVariable(0, "x", seed.X(), 0.1);
  minimizer->SetVariable(1, "y", seed.Y(), 0.1);
  minimizer->SetVariable(2, "z", seed.Z(), 0.1);

  bool success = minimizer->Minimize();
  if (!success) return false;

  const double* xs = minimizer->X();
  fittedVertex.SetXYZ(xs[0], xs[1], xs[2]);
  chi2Out = minimizer->MinValue();
  return true;
}

using namespace Pythia8;

int main(int argc, char* argv[]) {
  // Number of events (default 100k if not passed)
  int nEvents = 20000;
  if (argc > 1) nEvents = atoi(argv[1]);

  // Configure Pythia for pp collisions @ 13 TeV, b-quark production
  Pythia pythia;

  pythia.readString("Beams:idA = 2212");           // proton
  pythia.readString("Beams:idB = 2212");           // proton
  pythia.readString("Beams:eCM = 13600.");         // 13 TeV
  pythia.readString("Random:seed = 22");           // set seed
  pythia.readString("HardQCD:gg2bbbar = on");      // turn on gg->bb
  pythia.readString("HardQCD:qqbar2bbbar = on");   // turn on qqbar->bb
  // pythia.readString("PhaseSpace:pTHatMin = 40.");   // pT hat cut (GeV)

  // Vertex smearing for primary vertex (Gaussian)
  pythia.readString("Beams:allowVertexSpread = on");
  pythia.readString("Beams:sigmaVertexX = 0.01");         // mm (PV resolution)
  pythia.readString("Beams:sigmaVertexY = 0.01");         // mm
  pythia.readString("Beams:sigmaVertexZ = 0.035");         // mm

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
  TFile outFile("Simulation_Data_test.root", "RECREATE");
  TTree tree("Events", "B0 -> Kst Tau Tau production with vertices and uncertainties");

  // Kinematics
  Float_t ptB, etaB, phiB;
  Float_t mu_pt, mu_eta, mu_phi;
  Float_t mu_x, mu_y, mu_z;
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
  PVzErr = 0.035;  // mm (sigmaZ)

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
  tree.Branch("mu_pt",  &mu_pt,  "mu_pt/F");
  tree.Branch("mu_eta", &mu_eta, "mu_eta/F");
  tree.Branch("mu_phi", &mu_phi, "mu_phi/F");
  tree.Branch("mu_x",  &mu_x,  "mu_x/F");
  tree.Branch("mu_y",  &mu_y,  "mu_y/F");
  tree.Branch("mu_z",  &mu_z,  "mu_z/F");

  // Single-entry daughter kinematics
  Float_t kst_pt, kst_eta, kst_phi;
  Float_t tauPlus_pt, tauPlus_eta, tauPlus_phi;
  Float_t tauMinus_pt, tauMinus_eta, tauMinus_phi;
  Float_t m_tauMinus, m_tauPlus, m_kst;
  Float_t mT_tautau, m_tautau_coll;
  Float_t B0_t;


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
  tree.Branch("B0_t", &B0_t, "B0_t/F");
  Float_t IP_mu;
  tree.Branch("IP_mu", &IP_mu, "IP_mu/F");


  // Random number generators for measurement smearing and uncertainties
  std::mt19937 rng(42);
  std::normal_distribution<double> smearSVxy(0.0, 0.01);    // mm (SV spatial resolution)
  std::normal_distribution<double> smearSVz(0.0, 0.035);      // mm

  //stop when reaching goal of events
  int passed = 0;
  int generated = 0;
  const int targetPassed = 20000;
  bool done = false;

  // Event loop: find all B0 in the event record
  for (int iEvent = 0; iEvent < nEvents && !done; ++iEvent) {
    if (!pythia.next()) continue;
    ++generated;

    // look for a B⁰ with pT > 10 GeV
    bool keep = false;
    for (auto& p : pythia.event) {
      if ( (p.id()==511 || p.id()==-511) && p.pT() > 10.0 ) {
        keep = true;
        break;
      }
    }
    if (!keep) continue;


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

        /*std::cout << "[DEBUG] Found branches:"
          << " K*0="<<(kstar? "yes":"NO")
          << " tau+="<<(tauP? "yes":"NO")
          << " tau-="<<(tauM? "yes":"NO")<<"\n";*/
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
          if (nMuon == 1 /*&& leaves.size() >= 2*/) return "MU";
          // exactly three charged pions => 3π decay
          if (nPion == 3 /*&& leaves.size() == 4*/) return "3P";
          return "OTHER";
        };

        std::string typeP = classifyTau(tauPLeaves);
        std::string typeM = classifyTau(tauMLeaves);

        //std::cout << "[DEBUG] τ+ decay = " << typeP << ", τ- decay = " << typeM << "\n";

        // require exactly one MU and one 3P
        if (!((typeP=="MU" && typeM=="3P") ||
              (typeP=="3P"&& typeM=="MU")) ){
          //std::cout<<"[INFO] skipping event: not μ+3π\n";
          delete evtB;
          continue;
              }

        // find muon leaf and collect the muon
        const auto& muLeaves = (typeP=="MU" ? tauPLeaves : tauMLeaves);

        EvtParticle* muPart = nullptr;
        for (auto p : muLeaves) {
          if ( std::abs(EvtPDL::getStdHep(p->getId())) == 13 ) {
            muPart = p;
            break;
          }
        }
        // save muon kinematics and prod origin
        if (muPart) {
          EvtVector4R pos = muPart->get4Pos();
          mu_x = pos.get(1) + smearSVxy(rng);
          mu_y = pos.get(2) + smearSVxy(rng);
          mu_z = pos.get(3) + smearSVz(rng);
          EvtVector4R p4 = muPart->getP4Lab();
          TLorentzVector mu4(p4.get(1), p4.get(2), p4.get(3), p4.get(0));
          mu_pt  = mu4.Pt();
          mu_eta = mu4.Eta();
          mu_phi = mu4.Phi();
          if (mu_pt < 5.0) {
            delete evtB;
            continue;
          }
          // 3D impact parameter = | (muOrig – PV) × muDir |
          TVector3 PV(PVx, PVy, PVz);
          TVector3 muOrig(mu_x, mu_y, mu_z);
          TVector3 muDir = mu4.Vect().Unit();
          TVector3 d3 = muOrig - PV;
          IP_mu = d3.Cross(muDir).Mag();
        } else {
          mu_pt = mu_eta = mu_phi = -999.;
          mu_x = mu_y = mu_z = -999.;
          IP_mu = -999.;
        }


        // define a little helper to sum and collect only non‑neutrino four‑vectors:
        auto sumVisible = [&](const std::vector<EvtParticle*>& leaves) -> std::pair<TLorentzVector, std::vector<TLorentzVector>>{
          TLorentzVector sum(0,0,0,0);
          std::vector<TLorentzVector> visTracks;
          for(auto dau : leaves){
            int pdg = EvtPDL::getStdHep(dau->getId());
            //std::cout << "[DEBUG]   leaf PDG="<<pdg<<"\n";
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
        //TLorentzVector kstarReco = sumVisible(kstarLeaves).first;
        // build reco-vectors and grab visible tracks
        auto [kstarReco, kstarVis] = sumVisible(kstarLeaves);
        auto [tauMvisSum, tauMvis] = sumVisible(tauMLeaves);
        auto [tauPvisSum, tauPvis] = sumVisible(tauPLeaves);

        // require all visible tracks in CMS acceptance, eta cut
        bool allInEta = true;
        for (auto &v : kstarVis) if (std::abs(v.Eta()) > 2.5) { allInEta=false; break; }
        for (auto &v : tauPvis)   if (allInEta && std::abs(v.Eta()) > 2.5) { allInEta=false; break; }
        for (auto &v : tauMvis)    if (allInEta && std::abs(v.Eta()) > 2.5) { allInEta=false; break; }
        if (!allInEta) {
          delete evtB;
          continue;
        }

        double sigma_pt_Kst = kstarReco.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_Kst(kstarReco.Pt(), sigma_pt_Kst);
        kst_pt  = smearPt_Kst(rng);
        kst_eta = kstarReco.Eta();
        kst_phi = kstarReco.Phi();
        m_kst = kstarReco.M();

        // same for the tau branches
        //TLorentzVector tauMvisSum = sumVisible(tauMLeaves).first;
        //TLorentzVector tauPvisSum = sumVisible(tauPLeaves).first;

        double sigma_pt_tauP = tauPvisSum.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_tauP(tauPvisSum.Pt(), sigma_pt_tauP);
        tauPlus_pt  = smearPt_tauP(rng);
        tauPlus_eta = tauPvisSum.Eta();
        tauPlus_phi = tauPvisSum.Phi();
        m_tauPlus = tauPvisSum.M();
        //std::cout << "[DEBUG] mass of Tau Plus: " << m_tauPlus << "\n";

        // …and for the τ‑ branch:
        double sigma_pt_tauM = tauMvisSum.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_tauM(tauMvisSum.Pt(), sigma_pt_tauM);
        tauMinus_pt  = smearPt_tauM(rng);
        tauMinus_eta = tauMvisSum.Eta();
        tauMinus_phi = tauMvisSum.Phi();
        m_tauMinus = tauMvisSum.M();
        //std::cout << "[DEBUG] mass of Tau Minus: " << m_tauMinus << "\n";

        // fit vertex of reconstructed Kstar and tau tracks
        std::vector<Track> tracks;
        auto makeTrack = [&](const TLorentzVector& p4, const EvtParticle* particle) {
          // Use the particle's production vertex
          EvtVector4R v4 = particle->get4Pos();
          TVector3 origin(v4.get(1), v4.get(2), v4.get(3));
          // (Optional) smear by detector resolution:
          origin += TVector3(smearSVxy(rng), smearSVxy(rng), smearSVz(rng));
          TVector3 dir = p4.Vect().Unit();
          tracks.push_back(Track{origin, dir});
        };
        makeTrack(kstarReco, kstar);
        makeTrack(tauPvisSum, tauP);
        makeTrack(tauMvisSum, tauM);
        TVector3 vtx;
        float chi2;
        if (!FitVertex(tracks, 0.1, vtx, chi2)) continue;

        //int ndof = static_cast<int>(tracks.size())*2 - 3;     // 2 constraints per track minus 3 free coords
        //float chi2ndf = chi2 / ndof;
        SVx        = vtx.X();
        SVy        = vtx.Y();
        SVz        = vtx.Z();
        vertexChi2 = chi2;
        //std::cout << "[CHECK] Vertex Chi2 value: " << chi2 << std::endl;

        tree.Fill();

        // destroy the EvtGen particle to free memory:
        delete evtB;
        ++passed;
        if (passed >= targetPassed) {
          std::cout
            << "Generated " << generated
            << " events to save " << passed
            << " events.\n";
          done = true;
          break;
        }
      }
    }
  }

  if (passed < targetPassed) {
    std::cout
      << "Loop ended after " << generated
      << " generated events, but only "
      << passed << " passed cuts.\n";
  }

  std::cout << "Entries in tree: " << tree.GetEntries() << "\n";

  // Finalize
  pythia.stat();      // print summary to stdout
  outFile.Write();    // save TTree
  outFile.Close();

  return 0;
}
