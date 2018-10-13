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

	cv::Mat CreateGaussian1D(int len, float sigma);
	cv::Mat CreateGaussian2D(const cv::Size &sz, const float &sigma);
	cv::Mat GetGaussianSharpLabels(const cv::Size &sz, const float &sigma);
	cv::Mat CircShift(const cv::Mat &src, const cv::Size &V);
	cv::Mat hann(int len);
	cv::Mat get_subwindow(const cv::Mat &img, cv::Point pos, cv::Size sz);
	std::vector<cv::Mat> get_features(const cv::Mat &img, const cv::Mat &cos_window,const std::string &feature_type);
	std::vector<cv::Mat> get_hog(const cv::Mat &img);
	cv::Mat div_pointwise(const cv::Mat &x, const cv::Mat &y);
	cv::Mat mul_pointwise(const cv::Mat &x, const cv::Mat &y);
	cv::Point find_max(const cv::Mat&);

	std::vector<bbox> res;
protected:
	cv::Point target_pos;
	cv::Size target_sz;

	cv::Mat yf;
	cv::Mat cos_window;

};

class BACF :public Tracker {
public:
	BACF() {};
	~BACF() {};

	void init(bbox);
	bbox run(cv::Mat, int);
private:
	double learning_rate        = 0.013;
	double output_sigma_factor  = (double)(1. / 16);
	int    search_area_scale    = 5;
	int    filter_max_area      = 50 * 50;
	int    interpolate_response = 4;
	int    newton_iterations    = 50;
	int    number_of_sacles     = 5;
	int    cell_size            = 4;
	double scale_step           = 1.01;
};

class KCF :public Tracker {
public:
	KCF() {};
	~KCF() {};

	void init(bbox);
	bbox run(cv::Mat, int);

	cv::Mat Gaussian_kernel(const std::vector<cv::Mat> &xf,const std::vector<cv::Mat> &yf);

private:

	cv::Size window_size;

	cv::Mat model_alphaf;
	std::vector<cv::Mat> model_xf;

	bool resize_image = false;

	std::string feature_type  = "Hog";
	std::string kernel_type   = "Gaussian";

	float interp_factor       = 0.02;
	float kernel_sigma        = 0.5;
	float padding             = 1.5;
	float lambda              = 1e-4;
	float output_sigma_factor = 0.1;
	int   cell_size           = 4;
};

#endif // !TRACKER_H
