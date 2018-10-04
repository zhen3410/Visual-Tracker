#pragma once
#ifndef TRACKER_H
#define TRACKER_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include"Sequences.h"


class Tracker {
public:
	Tracker() {};
	~Tracker() {};
	virtual bbox run(cv::Mat,int) = 0;

private:

};

class KCF :public Tracker {
public:
	KCF() {};
	~KCF() {};

	void init(bbox);
	bbox run(cv::Mat, int);

	cv::Mat CreateGaussian1D(int len, float sigma);
	cv::Mat CreateGaussian2D(const cv::Size &sz,const float &sigma);
	cv::Mat GetGaussianSharpLabels(const cv::Size &sz,const float &sigma);
	cv::Mat CircShift(const cv::Mat &src, const cv::Size &V);
	cv::Mat hann(int len);
	cv::Mat get_subwindow(const cv::Mat &img, cv::Point pos, cv::Size sz);
	std::vector<cv::Mat> get_features(const cv::Mat &img, const cv::Mat &cos_window);
	std::vector<cv::Mat> get_hog(const cv::Mat &img);
	cv::Mat Gaussian_kernel(const std::vector<cv::Mat> &xf,const std::vector<cv::Mat> &yf);
	cv::Mat div_pointwise(const cv::Mat &x,const cv::Mat &y);
	cv::Mat mul_pointwise(const cv::Mat &x, const cv::Mat &y);
	cv::Point find_max(const cv::Mat&);

	std::vector<bbox> res;

private:

	cv::Point target_pos;
	cv::Size target_sz;
	cv::Size window_size;

	cv::Mat yf;
	cv::Mat cos_window;
	cv::Mat model_alphaf;
	std::vector<cv::Mat> model_xf;

	bool resize_image = false;


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
