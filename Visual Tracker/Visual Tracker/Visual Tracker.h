#pragma once
#ifndef VISUAL_TRACKER_H
#define VISUAL_TRACKER_H

#include<iostream>
#include<fstream>
using namespace std;

#include"Tracker.h"
#include"Sequences.h"

class Visual_Tracker {
public:
	Visual_Tracker() {};
	Visual_Tracker(Video* s) { seq = s; }
	vector<bbox> run(int ind);
private:
	Video * seq;
};

#endif // !VISUAL_TRACKER_H
