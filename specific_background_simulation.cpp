#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"
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

  std::mt19937 rng(41);
  std::normal_distribution<double> smearSVxy(0.01);
  std::normal_distribution<double> smearSVz(0.01);
  std::chi_squared_distribution<double> chi2Dist(6);

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
        double chi2ndf = chi2 / 6.0;  // Since DOF = 6
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
