
#include"stdafx.h"
#include<iostream>
#include<opencv2/opencv.hpp>
#include<core.hpp>
#include<imgproc.hpp>
#include<string>

#include"Tracker.h"
#include"fhog.h"

cv::Mat Tracker::CreateGaussian1D(int len, float sigma)
{
	cv::Mat label(len, 1, CV_32F);
	float* label_ptr = label.ptr<float>();
	float scale = -0.5 / (sigma*sigma);

	for (int i = 0; i < len; i++) {
		float x = i + 1 - floor(len / 2);
		float value = std::exp(scale*x*x);
		label_ptr[i] = value;
	}
	return label;
}

cv::Mat Tracker::CreateGaussian2D(const cv::Size &sz,const  float &sigma)
{
	cv::Mat a = CreateGaussian1D(sz.width, sigma);
	cv::Mat b = CreateGaussian1D(sz.height, sigma);

	cv::Mat lable = b * a.t();
	return lable;
}

cv::Mat Tracker::GetGaussianSharpLabels(const cv::Size &sz, const float &sigma)
{
	cv::Mat label = CreateGaussian2D(sz, sigma);

	cv::Size circ_V = cv::Size(-floor((float)sz.width / 2)+1, -floor((float)sz.height / 2) + 1);

	label = CircShift(label, circ_V);

	return label;
}

cv::Mat Tracker::CircShift(const cv::Mat &src, const cv::Size &V)
{
	cv::Mat res(src.size(), CV_32F);

	int h, w;
	if (V.width < 0) {
		w = -V.width;
	}
	else {
		w = src.cols - V.width;
	}
	if (V.height < 0) {
		h = -V.height;
	}
	else {
		h = src.rows - V.height;
	}

	cv::Mat a1(src, cv::Rect(0, 0, w, h));
	cv::Mat a2(src, cv::Rect(w, 0, src.cols - w, h));
	cv::Mat a3(src, cv::Rect(0, h, w, src.rows - h));
	cv::Mat a4(src, cv::Rect(w, h, src.cols - w, src.rows - h));

	cv::Mat temp1, temp2;
	cv::hconcat(a4, a3, temp1);
	cv::hconcat(a2, a1, temp2);
	cv::vconcat(temp1, temp2, res);


	return res;
}

cv::Mat Tracker::hann(int len)
{
	cv::Mat res(len, 1, CV_32F);

	float* ptr = res.ptr<float>();

	float PI = asin(1) * 2;

	for (int i = 0; i < len; i++) {
		ptr[i] = (1 - cos(2 * PI*i / (len - 1))) / 2.;
	}

	return res;
}

cv::Mat Tracker::get_subwindow(const cv::Mat &img, cv::Point pos, cv::Size sz)
{
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#define max(a,b)            (((a) > (b)) ? (a) : (b))
	cv::Point lefttop(pos.x - floor(sz.width / 2), pos.y - floor(sz.height / 2));
	cv::Point rightbottom(pos.x + sz.width - floor(sz.width / 2), pos.y + sz.height - floor(sz.height / 2));
	cv::Rect border(-min(lefttop.x, 0), -min(lefttop.y, 0), max(rightbottom.x - img.cols + 1, 0), max(rightbottom.y - img.rows + 1, 0));
	cv::Point r_lefttop(max(lefttop.x, 0), max(lefttop.y, 0));
	cv::Point r_rightbottom(min(rightbottom.x, img.cols), min(rightbottom.y, img.rows));
	cv::Rect r_rect(r_lefttop, r_rightbottom);

	cv::Mat res;
	img(r_rect).copyTo(res);
	cv::copyMakeBorder(res, res, border.y, border.height, border.x, border.width, cv::BORDER_REPLICATE);

#undef min(a,b)
#undef max(a,b)

	return res;
}

std::vector<cv::Mat> Tracker::get_features(const cv::Mat &img, const cv::Mat &cos_window,const std::string &feature_type)
{
	std::vector<cv::Mat> feature;

	if (feature_type == "Hog") {
		feature = get_hog(img);
	}
	else if (feature_type == "Gray") {
		feature.push_back(img.clone());
	}

	int dim = feature.size();
	std::vector<cv::Mat> res;
	for (int i = 0; i < dim; i++)
		res.push_back(feature[i].mul(cos_window));

	return res;
}

std::vector<cv::Mat> Tracker::get_hog(const cv::Mat &img)
{
	std::vector<cv::Mat> res;

	res = fhog(img);

	return res;
}

cv::Mat Tracker::div_pointwise(const cv::Mat &x,const cv::Mat &y)
{
	cv::Mat res;
	//std::vector<cv::Mat> plane1;
	//cv::Mat plane1[] = { cv::Mat_<float>(x),cv::Mat::zeros(x.size(),CV_32F) };
	cv::Mat plane1[2];
	cv::split(x, plane1);
	//std::vector<cv::Mat> plane2;
	//cv::Mat plane2[] = { cv::Mat_<float>(y),cv::Mat::zeros(y.size(),CV_32F) };
	cv::Mat plane2[2];
	cv::split(y, plane2);
	cv::Mat c2_d2 = plane2[0].mul(plane2[0]) + plane2[1].mul(plane2[1]);
	//cv::Mat res_t[] = { cv::Mat_<float>(y),cv::Mat::zeros(y.size(),CV_32F) };
	cv::Mat res_t[2];
	res_t[0] = plane1[0].mul(plane2[0]) + plane1[1].mul(plane2[1]);
	res_t[0] = res_t[0] / c2_d2;
	res_t[1] = plane1[1].mul(plane2[0]) - plane1[0].mul(plane2[1]);
	res_t[1] = res_t[1] / c2_d2;
	cv::merge(res_t, 2, res);

	return res;
}

cv::Mat Tracker::mul_pointwise(const cv::Mat & x, const cv::Mat & y)
{
	cv::Mat res;

	cv::Mat plane1[2];
	cv::split(x, plane1);
	cv::Mat plane2[2];
	cv::split(y, plane2);

	cv::Mat res_t[2];
	res_t[0] = plane1[0].mul(plane2[0]) - plane1[1].mul(plane2[1]);
	res_t[1] = plane1[0].mul(plane2[1]) + plane1[1].mul(plane2[0]);
	cv::merge(res_t, 2, res);

	return res;
}

cv::Point Tracker::find_max(const cv::Mat &x)
{
	cv::Point res;

	float max_x, max_y, max_v = 0;
	int m = x.cols, n = x.rows;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (x.at<float>(j,i) > max_v) {
				max_v = x.at<float>(j, i);
				max_x = i;
				max_y = j;
			}
		}
	}

	return cv::Point(max_x, max_y);
}

//****************************************************************************************************************************************
//KCF Algorithm programed by Zhang Zhen
//J.F. Henriques,R. Caseiro,P. Martins,and J. Batista, "High-Speed Tracking with Kernelized Correlation Filters," in Trans. on PAMI, 2015.
//KCF begin!

void KCF::init(bbox groundtruth)
{
	target_pos.x = (int)groundtruth.x;
	target_pos.y = (int)groundtruth.y;
	target_sz.width = (int)groundtruth.w;
	target_sz.height = (int)groundtruth.h;

	bool resize_image = (sqrt(target_sz.width*target_sz.height) >= 100);
	if (resize_image) {
		target_pos.x = floor(target_pos.x / 2.);
		target_pos.y = floor(target_pos.y / 2.);
		target_sz.width = floor(target_sz.width / 2.);
		target_sz.height = floor(target_sz.height / 2.);
	}

	window_size.width = floor(target_sz.width*(1 + padding));
	window_size.height = floor(target_sz.height*(1 + padding));

	float output_sigma = sqrt(target_sz.width*target_sz.height)*output_sigma_factor / cell_size;
	cv::Size use_sz;
	use_sz.width = window_size.width / cell_size;
	use_sz.height = window_size.height / cell_size;
	cv::Mat y = GetGaussianSharpLabels(use_sz, output_sigma);

	cv::dft(y, yf, cv::DFT_COMPLEX_OUTPUT);

	cos_window = hann(use_sz.height)*hann(use_sz.width).t();

}

bbox KCF::run(cv::Mat img, int frame)
{
	bbox result;

	cv::Mat img_o = img;

	if (resize_image)cv::resize(img, img, cv::Size(), 0.5, 0.5);

	if (img.channels() > 1) {
		cv::Mat img_temp = img;
		cv::cvtColor(img_temp, img, CV_BGR2GRAY);
	}

	if (frame > 0) {
		cv::Mat patch = get_subwindow(img, target_pos, window_size);

		std::vector<cv::Mat> x = get_features(patch, cos_window, feature_type);
		std::vector<cv::Mat> xf(x.size());
		for (int i = 0; i < x.size(); i++)
			cv::dft(x[i], xf[i], cv::DFT_COMPLEX_OUTPUT);
		cv::Mat kzf = Gaussian_kernel(xf, model_xf);
		cv::Mat responsef = mul_pointwise(model_alphaf, kzf);
		cv::Mat response;
		cv::idft(responsef, response, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

		cv::Point delta;
		cv::minMaxLoc(response, NULL, NULL, NULL, &delta);
		//delta = find_max(response);

		if ((delta.x + 1) > (x[0].cols / 2)) {
			delta.x -= x[0].cols;
		}
		if ((delta.y + 1) > (x[0].rows / 2)) {
			delta.y -= x[0].rows;
		}

		target_pos = target_pos + cell_size * (delta);

	}

	cv::Mat patch = get_subwindow(img, target_pos, window_size);

	std::vector<cv::Mat> x = get_features(patch, cos_window, feature_type);

	std::vector<cv::Mat> xf(x.size());
	for (int i = 0; i < x.size(); i++) {
		cv::dft(x[i], xf[i], cv::DFT_COMPLEX_OUTPUT);
	}

	cv::Mat kf = Gaussian_kernel(xf, xf);

	cv::Mat alpha_f = div_pointwise(yf, (kf + cv::Scalar(lambda, 0)));


	if (frame == 0) {
		model_alphaf = alpha_f;
		model_xf = xf;
	}
	else {
		model_alphaf = (1 - interp_factor)*model_alphaf + interp_factor * alpha_f;
		int len = model_xf.size();
		for (int i = 0; i < len; i++)
			model_xf[i] = (1 - interp_factor)*model_xf[i] + interp_factor * xf[i];
	}

	if (resize_image)result = (bbox(target_pos * 2, target_sz * 2));
	else result = (bbox(target_pos, target_sz));

	res.push_back(result);

	return result;
}

cv::Mat KCF::Gaussian_kernel(const std::vector<cv::Mat> &xf, const std::vector<cv::Mat> &yf)
{
	cv::Size sz = xf[0].size();
	int N = sz.width*sz.height;
	int dim = xf.size();
	float xx = 0, yy = 0;
	cv::Mat xyf, _xy, xy(xf[0].size(), CV_32FC1, cv::Scalar(0.0));
	for (int i = 0; i < dim; i++) {
		xx += cv::norm(xf[i])*cv::norm(xf[i]) / N;
		yy += cv::norm(yf[i])*cv::norm(yf[i]) / N;
		cv::mulSpectrums(xf[i], yf[i], xyf, 0, true);
		cv::idft(xyf, _xy, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		xy += _xy;
	}
	cv::Mat k, kf;
	exp(((-1 / (kernel_sigma * kernel_sigma))*cv::max(0, ((xx + yy - 2 * xy) / (N*dim)))), k);
	k.convertTo(k, CV_32FC1);
	cv::dft(k, kf, cv::DFT_COMPLEX_OUTPUT);
	return kf;
}

//KCF end!
//***************************************************************************************************************************************






//***************************************************************************************************************************************
//BACF Algorithm programed by Zhang Zhen
//H.K. Galoogahi, A. Fagg, S. Lucey, "Learning Background-Aware Correlation Filters for Visual Tracking," in Proc.of ICCV,2017.
//BACF begin!

void BACF::init(bbox groundtruth) {

	target_pos.x = (int)groundtruth.x;
	target_pos.y = (int)groundtruth.y;
	target_sz.width = groundtruth.w;
	target_sz.height = groundtruth.h;

	int search_area=(int)(target_sz.width/cell_size)*(target_sz.height/cell_size);

}

bbox BACF::run(cv::Mat img, int frame) {

	return bbox();
}
