#include<opencv2/opencv.hpp>


class bbox {
public:
	bbox(){}
	bbox(const cv::Point &pos, const cv::Size &sz) { x = pos.x; y = pos.y; h = sz.height; w = sz.width; }

	float x, y;
	float h, w;
};