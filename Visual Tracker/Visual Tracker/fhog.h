#pragma once
#include<vector>
#include<opencv2/opencv.hpp>

#include"gradientMex.h"

std::vector<cv::Mat> fhog(const cv::Mat &img, int use_hog = 2, int cell_size = 4, int n_orients = 9, int soft_bin = -1, float clip = 0.2) {
	int h = img.rows, w = img.cols, d = 1;
	bool full = true;
	if (h < 2 || w < 2) {
		std::cerr << "Img must be at least 2x2." << std::endl;
		return std::vector<cv::Mat>();
	}

	float* I = new float[h*w];
	const float* ptr_img = img.ptr<float>();
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			//I[i*h + j] = img.at<float>(j, i);
			I[i*h + j] = ptr_img[i*h + j];
		}
	}

	float *M = new float[h*w], *O = new float[h*w];
	gradMag(I, M, O, h, w, d, full);

	int n_chns = (use_hog == 0) ? n_orients : (use_hog == 1 ? n_orients * 4 : n_orients * 3 + 5);
	int hb = h / cell_size, wb = w / cell_size;

	float *H = new float[hb*wb*n_chns];
	memset(H, 0, hb*wb*n_chns * sizeof(float));

	if (use_hog == 0) {
		full = false;
		gradHist(M, O, H, h, w, cell_size, n_orients, soft_bin, full);
	}
	else if (use_hog == 1) {
		full = false;
		hog(M, O, H, h, w, cell_size, n_orients, soft_bin, full, clip);
	}
	else {
		fhog(M, O, H, h, w, cell_size, n_orients, soft_bin, clip);
	}

	std::vector<cv::Mat> res;
	int n_res_channels = (use_hog == 2) ? n_chns - 1 : n_chns;
	res.reserve(n_res_channels);
	for (int i = 0; i < n_res_channels; i++) {
		cv::Mat desc(hb, wb, CV_32F);
		for (int x = 0; x < wb; x++) {
			for (int y = 0; y < hb; y++) {
				desc.at<float>(y, x) = H[i*hb*wb + x * hb + y];
			}
		}
		res.push_back(desc.clone());
	}
	delete[] I;
	delete[] M;
	delete[] O;
	delete[] H;

	return res;
}