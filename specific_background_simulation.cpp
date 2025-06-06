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
  int nEvents = 3000;
  if (argc > 1) nEvents = atoi(argv[1]);

  // 1) Configure Pythia
  Pythia pythia;
  pythia.readString("Beams:idA = 2212");
  pythia.readString("Beams:idB = 2212");
  pythia.readString("Beams:eCM = 13600.");
  pythia.readString("Random:seed = 22");           // set seed
  pythia.readString("HardQCD:gg2bbbar = on");
  pythia.readString("HardQCD:qqbar2bbbar = on");
  pythia.readString("PhaseSpace:pTHatMin = 5.");
  pythia.readString("Beams:allowVertexSpread = on");
  pythia.readString("Beams:sigmaVertexX = 0.01");
  pythia.readString("Beams:sigmaVertexY = 0.01");
  pythia.readString("Beams:sigmaVertexZ = 0.025"); //in mm
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
  TFile outFile("Specific_Background_smeared.root", "RECREATE");
  TTree tree("Events", "Simulation of Specific Backgrounds");
  // Branch definitions
  Float_t ptB, etaB, phiB;
  Float_t PVx, PVy, PVz, PVxErr=0.01f, PVyErr=0.01f, PVzErr=0.025f;
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

  std::mt19937 rng(41);
  std::normal_distribution<double> smearSVxy(0, 0.01);
  std::normal_distribution<double> smearSVz(0, 0.025);
  std::chi_squared_distribution<double> chi2Dist(9);

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
        B0_t = properCTau;
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
/*
        // 5.d) Measurement smearing of SV
        SVx = SVx_true + smearSVxy(rng);
        SVy = SVy_true + smearSVxy(rng);
        SVz = SVz_true + smearSVz(rng);
        // 5.e) Assign uncertainties and chi2
        SVxErr = smearSVxy.stddev();
        SVyErr = smearSVxy.stddev();
        SVzErr = smearSVz.stddev();
        double chi2 = chi2Dist(rng);
        double chi2ndf = chi2/9;  // Since DOF = 9
        vertexChi2 = chi2ndf;*/

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

        TLorentzVector kstarReco = sumVisible(kstarLeaves).first;
        TLorentzVector tauReco = sumVisible(tauLeaves).first;
        TLorentzVector muReco = sumVisible(muLeaves).first;

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

        //int ndof = static_cast<int>(tracks.size())*2 - 3;     // 2 constraints per track minus 3 free coords
        //float chi2ndf = chi2 / ndof;
        SVx        = vtx.X();
        SVy        = vtx.Y();
        SVz        = vtx.Z();
        vertexChi2 = chi2;
        std::cout << "[CHECK] Vertex Chi2 value: " << chi2 << std::endl;

        /*bool gotKstar = false, gotTauplus = false, gotTauminus = false;

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
        }*/
        tree.Fill();
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
