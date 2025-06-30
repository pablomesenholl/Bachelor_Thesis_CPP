#include "EvtGen/EvtGen.hh"
#include "EvtGenBase/EvtRandomEngine.hh"
#include "EvtGenBase/EvtSimpleRandomEngine.hh"
#include "EvtGenExternal/EvtExternalGenList.hh"
#include "EvtGenModels/EvtAbsExternalGen.hh"
#include "EvtGenBase/EvtAbsRadCorr.hh"
#include "EvtGenBase/EvtDecayBase.hh"
#include <iostream>
#include "Tauola/Tauola.h"
#include "Pythia8/Pythia.h"
#include "TLorentzVector.h"
#include "TFile.h"
#include "TH1D.h"
#include "TCanvas.h"

using namespace Pythia8;

int main() {
    /*std::string decayFile = "/home/pablo/projects/bachelor_thesis_cpp/B0_Kst_tautau.dec";
    std::string pdtFile   = "/home/pablo/evtgen-new-install/share/EvtGen/evt.pdl";

    EvtRandomEngine* myRandomEngine = new EvtSimpleRandomEngine();

    EvtExternalGenList genList;
    EvtAbsRadCorr* photosEngine = genList.getPhotosModel();
    std::list<EvtDecayBase*> extraModels = genList.getListOfModels();

    try {
        // The fourth parameter is the radiation correction engine (optional), so we pass nullptr
        EvtGen myGenerator(decayFile, pdtFile, myRandomEngine, photosEngine, &extraModels);
        std::cout << "✅ EvtGen initialized successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception while initializing EvtGen: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "❌ Unknown error occurred while initializing EvtGen.\n";
        return 1;
    }

    delete myRandomEngine;
    return 0;*/
    // main.cpp


      // ——— Initialize Pythia ———
  Pythia pythia;
  pythia.readString("SoftQCD:nonDiffractive = on");
  pythia.readString("ParticleDecays:limitTau0 = off");
  // force only K*0→K+π−
  pythia.readString("313:onMode = off");
  pythia.readString("313:onIfMatch = 321 -211");
  pythia.readString("Beams:eCM = 13600.");
  pythia.init();

  // ——— Prepare ROOT histograms ———
  TH1D hTruth("hTruth",
              "inv M(K*0 to K+pi-) (truth);M (GeV);Entries",
              100, 0.0, 2.0);

  TH1D hMis_KK("hMis_KK",
               "inv M(K+K-) (misID);M (GeV);Entries",
               100, 0.0, 2.0);

  TH1D hMis_pipi("hMis_pipi",
                 "inv M(pi+pi-) (misID);M (GeV);Entries",
                 100, 0.0, 2.0);

  TH1D hMis_piK("hMis_piK",
                "inv M(pi+K-) (misID);M (GeV);Entries",
                100, 0.0, 2.0);

  const double mPi = 0.13957;   // GeV
  const double mK  = 0.493677;  // GeV

  const int nEvent = 1000;
  for (int iEvt = 0; iEvt < nEvent; ++iEvt) {
    if (!pythia.next()) continue;

    for (int i = 0; i < pythia.event.size(); ++i) {
      auto &p = pythia.event[i];
      // pick out the decayed K*0 resonance
      if (abs(p.id())!=313) continue;

      int d1 = p.daughter1(), d2 = p.daughter2();
      if (d1<0 || d2<0) continue;
      auto &c1 = pythia.event[d1];
      auto &c2 = pythia.event[d2];

      // identify which is K+ and which is π−
      Particle *k = nullptr, *pi = nullptr;
      if      (c1.id()==321 && c2.id()==-211) { k = &c1; pi = &c2; }
      else if (c2.id()==321 && c1.id()==-211) { k = &c2; pi = &c1; }
      else continue;

      // --- Truth-level mass ---
      TLorentzVector vK_t(  k->px(),  k->py(),  k->pz(),  k->e() );
      TLorentzVector vPi_t( pi->px(), pi->py(), pi->pz(), pi->e());
      hTruth.Fill( (vK_t + vPi_t).M() );

      // precompute momenta squared
      double p2k  = k->pAbs2();
      double p2pi = pi->pAbs2();

      // --- Mis-ID K+K-  (both masses = mK) ---
      TLorentzVector vK_KK(  k->px(),  k->py(),  k->pz(),
                             std::sqrt(p2k  + mK*mK) );
      TLorentzVector vPi_KK( pi->px(), pi->py(), pi->pz(),
                             std::sqrt(p2pi + mK*mK) );
      hMis_KK.Fill( (vK_KK + vPi_KK).M() );

      // --- Mis-ID π+π- (both masses = mPi) ---
      TLorentzVector vK_pp(  k->px(),  k->py(),  k->pz(),
                             std::sqrt(p2k  + mPi*mPi) );
      TLorentzVector vPi_pp( pi->px(), pi->py(), pi->pz(),
                             std::sqrt(p2pi + mPi*mPi) );
      hMis_pipi.Fill( (vK_pp + vPi_pp).M() );

      // --- Mis-ID π+K- (swap masses) ---
      TLorentzVector vK_pK(  k->px(),  k->py(),  k->pz(),
                             std::sqrt(p2k  + mPi*mPi) );
      TLorentzVector vPi_pK( pi->px(), pi->py(), pi->pz(),
                             std::sqrt(p2pi + mK *mK ) );
      hMis_piK.Fill( (vK_pK + vPi_pK).M() );
    }
  }

  // ——— Save everything ———
  TFile out("KstarStudies.root","RECREATE");
  hTruth.Write();
  hMis_KK.Write();
  hMis_pipi.Write();
  hMis_piK.Write();
  out.Close();

  // ——— Make overlay plot ———
  TCanvas c("c","Invariant Mass",800,600);
  hTruth.SetLineColor(kBlack);
  hTruth.SetLineWidth(2);
  hMis_KK.SetLineColor(kRed);
  hMis_pipi.SetLineColor(kBlue);
  hMis_piK.SetLineColor(kGreen+2);
  hMis_KK.SetLineWidth(2);
  hMis_pipi.SetLineWidth(2);
  hMis_piK.SetLineWidth(2);
    hTruth.SetStats(false);
    hMis_KK.SetStats(false);
    hMis_pipi.SetStats(false);
    hMis_piK.SetStats(false);
  hTruth.Draw();
  hMis_KK.Draw("SAME");
  hMis_pipi.Draw("SAME");
  hMis_piK.Draw("SAME");
  c.BuildLegend(0.6,0.7,0.9,0.9);
  c.SaveAs("InvariantMass_MisID.png");

  pythia.stat();
    std::cout
  << "Summary:\n"
  << "  K*0→K+π− truth fills:  " << hTruth.GetEntries()  << "\n"
  << "  misID K+K− fills:     " << hMis_KK.GetEntries() << "\n"
  << "  misID π+π− fills:     " << hMis_pipi.GetEntries()<< "\n"
  << "  misID π+K− fills:     " << hMis_piK.GetEntries() << "\n";
  return 0;
}
