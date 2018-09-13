#pragma once
#ifndef TRACKER_H
#define TRACKER_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include"Sequences.h"

class Tracker {
public:
	Tracker() {};
	virtual void run(Video *seq) const = 0;
private:

};

class KCF :public Tracker {
public:
	KCF() {};
	void run(Video *seq)const;

	cv::Mat CreateGaussian1D(int len, float sigma)const;
	cv::Mat CreateGaussian2D(cv::Size sz, float sigma)const;
	cv::Mat GetGaussianSharpLabels(cv::Size sz, float sigma)const;
	cv::Mat CircShift(cv::Mat src, cv::Size V)const;
	cv::Mat hann(int len)const;
	cv::Mat get_subwindow(cv::Mat img, cv::Point pos, cv::Size sz)const;
	std::vector<cv::Mat> get_features(cv::Mat img, cv::Mat cos_window)const;
	std::vector<cv::Mat> get_hog(cv::Mat img)const;
	cv::Mat Gaussian_kernel(std::vector<cv::Mat> xf, std::vector<cv::Mat> yf)const;
	cv::Mat div_pointwise(cv::Mat x,cv::Mat y)const;


private:
	std::string kernel_type = "Gaussian";
	std::string feature_type = "Hog";

	float interp_factor = 0.02;
	float kernel_sigma = 0.5;
	float padding = 1.5;
	float lambda = 1e-4;
	float output_sigma_factor = 0.1;
	int cell_size = 4;
};

#endif // !TRACKER_H
