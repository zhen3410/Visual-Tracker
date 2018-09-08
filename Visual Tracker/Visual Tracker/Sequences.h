#pragma once
#include<string>
#include<vector>
#include"bbox.h"


class Video {
public:
	Video() {}
	Video(std::string _path) { path = _path; }
	void init();

	int length;
	std::string name;
	std::string path;
	std::vector<std::string> img_path;

	std::vector<bbox> ground_truth;

};