#include "TROOT.h"
#include "TH1F.h"
#include "TF1.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TFile.h"
#include "TMath.h"
#include "TLine.h"
#include "TText.h"
#include "TTree.h"
#include "TCanvas.h"
#include <fstream>
#include <sstream>
#include <string>
#include "langaus.h"


void fitLanGau(TH1F& hist){
  // Setting fit range and start values
  Double_t fr[2];
  Double_t sv[4], pllo[4], plhi[4], fpe[4]; Double_t*fp = new Double_t[4];
  
  fr[0]=0.5*hist.GetMean();
  fr[1]=2.0*hist.GetMean();

  //parameters to tune
  pllo[0]=1.0; pllo[1]=10.0; pllo[2]=200.0; pllo[3]=0.0001;
  plhi[0]=100.0; plhi[1]=50.0; plhi[2]=250000.0; plhi[3]= 10.0;
  sv[0]=75.0; sv[1]=50.0; sv[2]=10000.0; sv[3]=30.0;//

  Double_t chisqr;
  Int_t    ndf;

  TF1 *fitsnr = langaufit(&hist,fr,sv,pllo,plhi,fp,fpe,&chisqr,&ndf);

  Double_t SNRPeak, SNRFWHM;
  //    langaupro(fp,SNRPeak,SNRFWHM);
  fitsnr->Draw("lsame");
  fitsnr->SetLineWidth(2);
  fitsnr->GetXaxis()->SetTitle("ToT counts");
  fitsnr->GetYaxis()->SetTitle("# of events");

  TText* t = new TText(.3,.75,Form("MPV %.1f %.1f", fp[1], fpe[1]) ) ;
  t->SetNDC(kTRUE) ;
  t->Draw("same") ;
}


void example(){

  //TFile* file = TFile::Open("c1.root");
  //TTree* tree = (TTree*)file->Get("c1");
  TH1F* h = new TH1F("photons","",100,0,500);

  ifstream infile ("photonnums.csv");
  float a;
  while(infile >> a){
    //cout << a << endl;
    //if(a!=nan){
    h->Fill(a);
  }

  //TTree* file = new TTree("t", "Number of photons ()");
  //file->ReadFile("photonnums.csv","gamma");
  TFile* ofile = TFile::Open("/user/marjoh/masterproject/MasterProjectFitting/example_output.root","recreate");  
  ////file->Draw("gamma","");

  //TH1F* h = new TH1F("photons","",500,0,500);
  
  //h = file.c1;
  //h->Fill(datalist);
  
  TCanvas* c = new TCanvas("gamma","",600,600);
  c->cd();
  h->Draw();
  h->GetXaxis()->SetRange(2,500);
  fitLanGau(*h);
  h->GetXaxis()->SetRange(1,500);
  
  ofile->cd();
  h->Write();
  c->Write();
  //tfile->Close();
  ofile->Close();
}
