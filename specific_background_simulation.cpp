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
  // Number of events
  int nEvents = 1000000;
  if (argc > 1) nEvents = atoi(argv[1]);

  // 1) Configure Pythia
  Pythia pythia;
  pythia.readString("Beams:idA = 2212");
  pythia.readString("Beams:idB = 2212");
  pythia.readString("Beams:eCM = 13600.");
  pythia.readString("Random:seed = 22");           // set seed
  pythia.readString("HardQCD:gg2bbbar = on");
  pythia.readString("HardQCD:qqbar2bbbar = on");
  pythia.readString("PhaseSpace:pTHatMin = 40.");
  pythia.readString("Beams:allowVertexSpread = on");
  pythia.readString("Beams:sigmaVertexX = 0.01");
  pythia.readString("Beams:sigmaVertexY = 0.01");
  pythia.readString("Beams:sigmaVertexZ = 0.035"); //in mm
  pythia.readString("ParticleDecays:limitTau0 = on");
  pythia.readString("511:mayDecay = off");
  pythia.readString("-511:mayDecay = off");
  pythia.init();

  // Initialize EvtGen
  const char* decayFile = "/home/pablo/projects/bachelor_thesis_cpp/Specific_background.dec";
  const char* pdlFile   = "/home/pablo/evtgen-install/share/EvtGen/evt.pdl";

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
  // Now construct EvtGen with custom decay files, PDG table, RNG, externals
  EvtGen* evtgen = new EvtGen(
    std::string(decayFile),
    std::string(pdlFile),
    randEng, photos, &externals, 1, false
  );

  // 3) Output file and tree
  TFile outFile("Specific_Background_Final.root", "RECREATE");
  TTree tree("Events", "Simulation of Specific Backgrounds");
  // Branch definitions
  Float_t ptB, etaB, phiB;
  Float_t PVx, PVy, PVz, PVxErr=0.01f, PVyErr=0.01f, PVzErr=0.035f;
  Float_t SVx, SVy, SVz, SVxErr, SVyErr, SVzErr, vertexChi2;
  Float_t kst_pt, kst_eta, kst_phi;
  Float_t tauPlus_pt, tauPlus_eta, tauPlus_phi;
  Float_t tauMinus_pt, tauMinus_eta, tauMinus_phi;
  Float_t m_tauPlus, m_tauMinus, m_kst;
  Float_t B0_t;
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
  tree.Branch("B0_t", &B0_t, "B0_t/F");
  Float_t mu_pt, mu_eta, mu_phi;
  Float_t mu_x, mu_y, mu_z;
  tree.Branch("mu_pt",    &mu_pt,    "mu_pt/F");
  tree.Branch("mu_eta",   &mu_eta,   "mu_eta/F");
  tree.Branch("mu_phi",   &mu_phi,   "mu_phi/F");
  tree.Branch("mu_x",    &mu_x,    "mu_x/F");
  tree.Branch("mu_y",    &mu_y,    "mu_y/F");
  tree.Branch("mu_z",    &mu_z,    "mu_z/F");
  Float_t IP_mu;
  tree.Branch("IP_mu", &IP_mu, "IP_mu/F");

  std::mt19937 rng(41);
  std::normal_distribution<double> smearSVxy(0, 0.01);
  std::normal_distribution<double> smearSVz(0, 0.035);

  //stop when reaching goal of events
  int passed = 0;
  int generated = 0;
  const int targetPassed = 10000;
  bool done = false;

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

        // EvtGen decay
        EvtVector4R  mom( p.e(), p.px(), p.py(), p.pz() );
        EvtParticle* evtB = EvtParticleFactory::particleFactory(
                      EvtPDL::evtIdFromStdHep(p.id()), mom );
        evtgen->generateDecay(evtB);

        EvtParticle* kstar = nullptr;
        EvtParticle* tau  = nullptr;
        EvtParticle* mu = nullptr;
        std::function<void(EvtParticle*)> findNode = [&](EvtParticle* node) {
          if (!node) return;
          for(int i=0; i<node->getNDaug(); ++i){
            EvtParticle* d = node->getDaug(i);
            EvtId id = d->getId();
            if      (id == EvtPDL::getId("K*0") || id == EvtPDL::getId("anti-K*0") ) kstar = d;
            else if (id == EvtPDL::getId("tau+" ) || id == EvtPDL::getId( "tau-")    ) tau  = d;
            else if (id == EvtPDL::getId("mu-" ) || id == EvtPDL::getId("mu+")   ) mu  = d;
            else findNode(d);
          }
        };

        findNode(evtB);

        auto collectLeaves = [&](EvtParticle* node, std::vector<EvtParticle*>& out){
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

        std::vector<EvtParticle*> kstarLeaves, tauLeaves, muLeaves;
        if (kstar) collectLeaves(kstar, kstarLeaves);
        if (tau) collectLeaves(tau,  tauLeaves);
        if (mu) collectLeaves(mu,  muLeaves);

        // pick up muon and save its kinematics
        EvtParticle* muPart = nullptr;
        for (auto p : muLeaves) {
          if (std::abs(EvtPDL::getStdHep(p->getId())) == 13) {
            muPart = p;
            break;
          }
        }
        if (muPart) {
          auto pos = muPart->get4Pos();
          mu_x = pos.get(1) + smearSVxy(rng);
          mu_y = pos.get(2) + smearSVxy(rng);
          mu_z = pos.get(3) + smearSVz (rng);
          auto p4 = muPart->getP4Lab();
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

        auto [kstarReco, kstarVis] = sumVisible(kstarLeaves);
        auto [tauReco,   tauVis]   = sumVisible(tauLeaves);
        auto [muReco,    muVis]    = sumVisible(muLeaves);

        //eta cut on all visible particles, CMS realistic
        bool allInEta = true;
        for (auto &v : kstarVis) if (std::abs(v.Eta()) > 2.5) { allInEta=false; break; }
        for (auto &v : tauVis)   if (allInEta && std::abs(v.Eta()) > 2.5) { allInEta=false; break; }
        for (auto &v : muVis)    if (allInEta && std::abs(v.Eta()) > 2.5) { allInEta=false; break; }
        if (!allInEta) {
          delete evtB;
          continue;
        }

        // relative pt error for low pt << 100GeV
        double sigma_pt_rel = 0.007;

        // build reco‑K* vector and fill:
        double sigma_pt_Kst = kstarReco.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_Kst(kstarReco.Pt(), sigma_pt_Kst);
        kst_pt  = smearPt_Kst(rng);
        kst_eta = kstarReco.Eta();
        kst_phi = kstarReco.Phi();
        m_kst = kstarReco.M();

        double sigma_pt_tauP = tauReco.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_tauP(tauReco.Pt(), sigma_pt_tauP);
        tauPlus_pt  = smearPt_tauP(rng);
        tauPlus_eta = tauReco.Eta();
        tauPlus_phi = tauReco.Phi();
        m_tauPlus = tauReco.M();
        //std::cout << "[DEBUG] mass of Tau Plus: " << m_tauPlus << "\n";

        // …and for the τ‑ branch:
        double sigma_pt_tauM = muReco.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_tauM(muReco.Pt(), sigma_pt_tauM);
        tauMinus_pt  = smearPt_tauM(rng);
        tauMinus_eta = muReco.Eta();
        tauMinus_phi = muReco.Phi();
        m_tauMinus = muReco.M();
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
        makeTrack(tauReco, tau);
        makeTrack(muReco, mu);
        TVector3 vtx;
        float chi2;
        if (!FitVertex(tracks, 0.1, vtx, chi2)) continue;

        SVx        = vtx.X();
        SVy        = vtx.Y();
        SVz        = vtx.Z();
        vertexChi2 = chi2;
        //std::cout << "[CHECK] Vertex Chi2 value: " << chi2 << std::endl;

        tree.Fill();
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

  std::cout << "Entries: " << tree.GetEntries() << std::endl;
  pythia.stat();
  outFile.Write();
  outFile.Close();
  return 0;
}
