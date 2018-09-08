#include"stdafx.h"
#include"Sequences.h"

#include<opencv2/opencv.hpp>
#include<Windows.h>
#include<iostream>
#include<fstream>
#include<sstream>

void Video::init() {

	HANDLE dir;
	WIN32_FIND_DATA file_data;

	if ((dir = FindFirstFile(((path + "/*").c_str()), &file_data)) == INVALID_HANDLE_VALUE) {
		std::cout << "No Images!!" << std::endl;
	}
	do {
		std::string filename = file_data.cFileName;
		//std::cout << filename << std::endl;
		bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
		if (filename[0] == '.')continue;
		if (is_directory)continue;
		std::size_t file_ex_pos = filename.find_last_of(".");
		std::string file_ex = filename.substr(file_ex_pos + 1);
		if (file_ex == "jpg" || file_ex == "png" || file_ex == "bmp") {
			std::string file_path = path + "/" + filename;
			img_path.push_back(file_path);
		}
		else if (file_ex == "txt") {
			std::string file_path = path + "/" + filename;
			std::ifstream in(file_path);
			std::string line;
			while (getline(in, line)) {
				std::vector<float> box;

				int len = line.size();
				float temp = 0;
				int temp2 = 1;
				bool flag = false;
				for (int i = 0; i <= len; i++) {
					if (line[i] == ',' || i == len) {
						box.push_back(temp);
						temp = 0;
						temp2 = 0;
						flag = false;
						continue;
					}
					else if (line[i] == '.') {
						flag = true;
					}
					if (flag) {
						temp2 /= 10;
						temp += (line[i] - '0')*temp2;
					}
					else {
						temp = temp * 10 + line[i] - '0';
					}
				}

				if (box.size() == 4) {
					bbox temp;
					temp.x = box[0];
					temp.y = box[1];
					temp.w = box[0] + box[2] / 2;
					temp.h = box[1] + box[3] / 2;
					ground_truth.push_back(temp);
				}
				else if (box.size() == 8) {
					bbox temp;
					temp.x = (box[0] + box[2] + box[4] + box[6]) / 4;
					temp.y = (box[1] + box[3] + box[5] + box[7]) / 4;
					float x1 = min(min(min(box[0], box[2]), box[4]), box[6]);
					float x2 = max(max(max(box[0], box[2]), box[4]), box[6]);
					float y1 = min(min(min(box[1], box[3]), box[5]), box[7]);
					float y2 = max(max(max(box[1], box[3]), box[5]), box[7]);
					float A1 = sqrt(pow((box[0] - box[2]), 2) + pow((box[1] - box[3]), 2))*sqrt(pow((box[2] - box[4]), 2) + pow((box[3] - box[5]), 2));
					float A2 = (x2 - x1)*(y2 - y1);
					float s = sqrt(A1 / A2);
					temp.w = s * (x2 - x1) + 1;
					temp.h = s * (y2 - y1) + 1;
					ground_truth.push_back(temp);
				}
			}
		}

	} while (FindNextFile(dir, &file_data));

	length = img_path.size();
	/*
	std::vector<cv::String> fn;
	glob(cv::String(path+"/*.jpg"),fn,false);
	*/

}