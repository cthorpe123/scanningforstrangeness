#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TLegend.h>
#include <TAxis.h>
#include <iostream>

void plot_loss(const char* file_path) 
{
    TFile *file = TFile::Open(file_path);
    if (!file || file->IsZombie()) 
    {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return;
    }

    TTree *tree = (TTree*)file->Get("metrics");
    if (!tree) 
    {
        std::cerr << "Error retrieving 'metrics' tree from file." << std::endl;
        file->Close();
        return;
    }

    std::cout << "Available branches in the TTree 'metrics':" << std::endl;
    TObjArray *branches = tree->GetListOfBranches();
    for (int i = 0; i < branches->GetEntries(); ++i) 
    {
        TBranch *branch = (TBranch*)branches->At(i);
        std::cout << branch->GetName() << std::endl;
    }

    double train_loss, train_loss_std, valid_signature_loss, valid_signature_loss_std;
    double valid_background_loss, valid_background_loss_std;
    double learning_rate;
    double sig_acc_mean, sig_acc_std, sig_prec_mean, sig_prec_std, sig_rec_mean, sig_rec_std;
    double bg_acc_mean, bg_acc_std, bg_prec_mean, bg_prec_std, bg_rec_mean, bg_rec_std;
    
    tree->SetBranchAddress("train_loss", &train_loss);
    tree->SetBranchAddress("train_loss_std", &train_loss_std);
    tree->SetBranchAddress("valid_signature_loss", &valid_signature_loss);
    tree->SetBranchAddress("valid_signature_loss_std", &valid_signature_loss_std);
    tree->SetBranchAddress("valid_background_loss", &valid_background_loss);
    tree->SetBranchAddress("valid_background_loss_std", &valid_background_loss_std);
    tree->SetBranchAddress("learning_rate", &learning_rate);
    
    tree->SetBranchAddress("signature_metrics_accuracy_mean", &sig_acc_mean);
    tree->SetBranchAddress("signature_metrics_accuracy_std", &sig_acc_std);
    tree->SetBranchAddress("signature_metrics_precision_mean", &sig_prec_mean);
    tree->SetBranchAddress("signature_metrics_precision_std", &sig_prec_std);
    tree->SetBranchAddress("signature_metrics_recall_mean", &sig_rec_mean);
    tree->SetBranchAddress("signature_metrics_recall_std", &sig_rec_std);
    
    tree->SetBranchAddress("background_metrics_accuracy_mean", &bg_acc_mean);
    tree->SetBranchAddress("background_metrics_accuracy_std", &bg_acc_std);
    tree->SetBranchAddress("background_metrics_precision_mean", &bg_prec_mean);
    tree->SetBranchAddress("background_metrics_precision_std", &bg_prec_std);
    tree->SetBranchAddress("background_metrics_recall_mean", &bg_rec_mean);
    tree->SetBranchAddress("background_metrics_recall_std", &bg_rec_std);

    std::vector<double> epochs, train_losses, train_loss_errors, valid_signature_losses, valid_signature_errors, valid_background_losses, valid_background_errors;
    std::vector<double> sig_accuracies, sig_acc_errors, sig_precisions, sig_prec_errors, sig_recalls, sig_rec_errors;
    std::vector<double> bg_accuracies, bg_acc_errors, bg_precisions, bg_prec_errors, bg_recalls, bg_rec_errors;
    std::vector<double> learning_rates;

    Long64_t n_entries = tree->GetEntries();
    for (Long64_t i = 0; i < n_entries; i++) 
    {
        tree->GetEntry(i);
        epochs.push_back(i + 1);
        train_losses.push_back(train_loss);
        train_loss_errors.push_back(train_loss_std);
        valid_signature_losses.push_back(valid_signature_loss);
        valid_signature_errors.push_back(valid_signature_loss_std);
        valid_background_losses.push_back(valid_background_loss);
        valid_background_errors.push_back(valid_background_loss_std);
        learning_rates.push_back(learning_rate);

        sig_accuracies.push_back(sig_acc_mean);
        sig_acc_errors.push_back(sig_acc_std);
        sig_precisions.push_back(sig_prec_mean);
        sig_prec_errors.push_back(sig_prec_std);
        sig_recalls.push_back(sig_rec_mean);
        sig_rec_errors.push_back(sig_rec_std);
        
        bg_accuracies.push_back(bg_acc_mean);
        bg_acc_errors.push_back(bg_acc_std);
        bg_precisions.push_back(bg_prec_mean);
        bg_prec_errors.push_back(bg_prec_std);
        bg_recalls.push_back(bg_rec_mean);
        bg_rec_errors.push_back(bg_rec_std);

        std::cout << "Epoch " << (i + 1) << ":\n";
        std::cout << "  Train Loss: " << train_loss << " ± " << train_loss_std << "\n";
        std::cout << "  Valid Signature Loss: " << valid_signature_loss << " ± " << valid_signature_loss_std << "\n";
        std::cout << "  Valid Background Loss: " << valid_background_loss << " ± " << valid_background_loss_std << "\n";
        std::cout << "  Learning Rate: " << learning_rate << "\n";
        std::cout << "  Sig. Accuracy: " << sig_acc_mean << " ± " << sig_acc_std << ", Precision: " << sig_prec_mean << " ± " << sig_prec_std << ", Recall: " << sig_rec_mean << " ± " << sig_rec_std << "\n";
        std::cout << "  Bg. Accuracy: " << bg_acc_mean << " ± " << bg_acc_std << ", Precision: " << bg_prec_mean << " ± " << bg_prec_std << ", Recall: " << bg_rec_mean << " ± " << bg_rec_std << "\n";
    }

    TCanvas *c1 = new TCanvas("c1", "Loss and Learning Rate Over Epochs", 900, 700);
    c1->SetLogy();

    TGraphErrors *train_loss_graph = new TGraphErrors(epochs.size(), &epochs[0], &train_losses[0], 0, &train_loss_errors[0]);
    TGraphErrors *valid_signature_graph = new TGraphErrors(epochs.size(), &epochs[0], &valid_signature_losses[0], 0, &valid_signature_errors[0]);
    TGraphErrors *valid_background_graph = new TGraphErrors(epochs.size(), &epochs[0], &valid_background_losses[0], 0, &valid_background_errors[0]);
    TGraph *learning_rate_graph = new TGraph(epochs.size(), &epochs[0], &learning_rates[0]);

    train_loss_graph->SetLineColor(kBlue + 1);
    train_loss_graph->SetMarkerColor(kBlue + 1);
    train_loss_graph->SetMarkerStyle(21);
    train_loss_graph->SetLineWidth(2);
    train_loss_graph->SetMinimum(0.1);
    train_loss_graph->SetMaximum(100);

    valid_signature_graph->SetLineColor(kRed + 1);
    valid_signature_graph->SetMarkerColor(kRed + 1);
    valid_signature_graph->SetMarkerStyle(22);
    valid_signature_graph->SetLineWidth(2);

    valid_background_graph->SetLineColor(kGreen + 2);
    valid_background_graph->SetMarkerColor(kGreen + 2);
    valid_background_graph->SetMarkerStyle(23);
    valid_background_graph->SetLineWidth(2);

    learning_rate_graph->SetLineColor(kBlack);
    learning_rate_graph->SetMarkerColor(kBlack);
    learning_rate_graph->SetMarkerStyle(24);
    learning_rate_graph->SetLineWidth(2);

    train_loss_graph->SetTitle("");
    train_loss_graph->GetXaxis()->SetTitle("Epoch");
    train_loss_graph->GetYaxis()->SetTitle("Loss");
    train_loss_graph->GetXaxis()->SetLimits(0, 13);
    train_loss_graph->GetXaxis()->SetNdivisions(13, 0, 0, false);
    train_loss_graph->GetXaxis()->SetLabelSize(0.03);
    train_loss_graph->GetYaxis()->SetLabelSize(0.03);
    train_loss_graph->GetYaxis()->SetTitleSize(0.04);
    train_loss_graph->Draw("ALP");

    valid_signature_graph->Draw("LP SAME");
    valid_background_graph->Draw("LP SAME");

    TLegend *legend1 = new TLegend(0.7, 0.7, 0.88, 0.88);
    legend1->SetBorderSize(0);
    legend1->SetFillStyle(0);
    legend1->AddEntry(train_loss_graph, "Training Loss", "lep");
    legend1->AddEntry(valid_signature_graph, "Validation Signature Loss", "lep");
    legend1->AddEntry(valid_background_graph, "Validation Background Loss", "lep");
    legend1->AddEntry(learning_rate_graph, "Learning Rate", "lep");
    legend1->Draw();

    c1->SaveAs("loss_learning_rate.png");

    file->Close();
    delete file;
}
