
#include"stdafx.h"
#include<iostream>
#include<opencv2/opencv.hpp>
#include<core.hpp>
#include<imgproc.hpp>
#include<string>


#include"Tracker.h"
#include"fhog.h"

void KCF::run(Video *seq)const
{
	//std::cout << "test complete!" << std::endl;

	cv::Point target_pos;
	target_pos.x = seq->ground_truth[0].x;
	target_pos.y = seq->ground_truth[0].y;
	cv::Size target_sz;
	target_sz.width = seq->ground_truth[0].w;
	target_sz.height = seq->ground_truth[0].h;

	cv::Size window_size;
	window_size.width = target_sz.width*padding;
	window_size.height = target_sz.height*padding;

	float output_sigma = sqrt(target_sz.width*target_sz.height)*output_sigma_factor / cell_size;
	cv::Size use_sz;
	use_sz.width = window_size.width / cell_size;
	use_sz.height = window_size.height / cell_size;
	cv::Mat y = CreateGaussian2D(use_sz, output_sigma);
	cv::Mat yf;
	cv::dft(y, yf, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat cos_window(use_sz, CV_32F);
	cos_window = hann(use_sz.width)*hann(use_sz.height).t();

	int video_len = seq->length;
	for (int frame = 0; frame < video_len; frame++) {
		cv::Mat img = cv::imread(seq->img_path[frame]);

		if (img.dims > 1) {
			cv::cvtColor(img, img, CV_RGB2GRAY);
		}
		if (frame > 0) {

		}

		cv::Mat patch = get_subwindow(img, target_pos, window_size);

		std::vector<cv::Mat> x = get_features(patch, cos_window);
		std::vector<cv::Mat> xf(x.size());
		for (int i = 0; i < x.size(); i++) {
			cv::dft(x[i], xf[i], cv::DFT_COMPLEX_OUTPUT);
		}

		cv::Mat kf = Gaussian_kernel(xf, xf);


	}
}

cv::Mat KCF::CreateGaussian1D(int len, float sigma)const
{
	cv::Mat label(len, 1, CV_32F);
	float* label_ptr = label.ptr<float>();
	double scale = -0.5 / (sigma*sigma);

	for (int i = 0; i < len; i++) {
		double x = i + 1 - floor(len / 2);
		double value = std::exp(scale*x*x);
		label_ptr[i] = value;
	}
	delete label_ptr;
	return label;
}

cv::Mat KCF::CreateGaussian2D(cv::Size sz, float sigma)const
{
	cv::Mat a = CreateGaussian1D(sz.width, sigma);
	cv::Mat b = CreateGaussian1D(sz.height, sigma);

	cv::Mat lable = a * b.t();
	return lable;
}

cv::Mat KCF::GetGaussianSharpLabels(cv::Size sz, float sigma) const
{
	cv::Mat label = CreateGaussian2D(sz, sigma);

	cv::Size circ_V = cv::Size(-(floor(sz.width / 2) + 1), -(floor(sz.height / 2) + 1));

	label = CircShift(label, circ_V);

	return label;
}

cv::Mat KCF::CircShift(cv::Mat src, cv::Size V) const
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
	cv::Mat a4(src, cv::Rect(w, h, src.cols - w, src.cols - h));

	cv::Mat temp1, temp2;
	cv::hconcat(a4, a3, temp1);
	cv::hconcat(a2, a1, temp2);
	cv::vconcat(temp1, temp2, res);


	return res;
}

cv::Mat KCF::hann(int len) const
{
	cv::Mat res(len, 1, CV_32F);

	float* ptr = res.ptr<float>();

	float PI = asin(1) * 2;

	for (int i = 0; i < len; i++) {
		ptr[i] = (1 - cos(2 * PI*i) / (len - 1)) / 2.;
	}
	delete ptr;

	return res;
}

cv::Mat KCF::get_subwindow(cv::Mat img, cv::Point pos, cv::Size sz) const
{
	cv::Mat res(img, cv::Rect(pos, sz));

	return res;
}

std::vector<cv::Mat> KCF::get_features(cv::Mat img, cv::Mat cos_window) const
{
	std::vector<cv::Mat> feature;

	if (feature_type == "Hog") {
		feature = get_hog(img);
	}
	else if (feature_type == "Gray") {
		feature.push_back(img.clone());
	}

	return feature;
}

std::vector<cv::Mat> KCF::get_hog(cv::Mat img) const
{
	std::vector<cv::Mat> res;

	res = fhog(img);

	return cv::Mat();
}

cv::Mat KCF::Gaussian_kernel(std::vector<cv::Mat> xf, std::vector<cv::Mat> yf) const
{
	cv::Size sz = xf[0].size();
	int N = sz.width*sz.height;
	int dim = xf.size();
	double xx = 0, yy = 0;
	cv::Mat xyf, _xy, xy;
	for (int i = 0; i < dim; i++) {
		xx += cv::norm(xf[i])*cv::norm(xf[i]) / N;
		yy += cv::norm(yf[i])*cv::norm(yf[i]) / N;
		cv::mulSpectrums(xf[i], xf[i], xyf, 0, true);
		cv::idft(xyf, _xy, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		xy += _xy;
	}
	cv::Mat k, kf;
	exp((-1 / kernel_sigma * kernel_sigma*max(0, (xx + yy - 2 * xy) / (N*dim))), k);
	cv::dft(k, kf, cv::DFT_COMPLEX_OUTPUT);
	return kf;
}

cv::Mat KCF::div_pointwise(cv::Mat x, cv::Mat y) const
{
	return cv::Mat();
}
