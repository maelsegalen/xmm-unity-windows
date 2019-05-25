#pragma once
//
//  xmmTrainingDataTools.h
//  XmmEngine
//
//  Created by Joseph Larralde on 19/01/17.
//  Copyright © 2017 IRCAM. All rights reserved.
//

#ifndef xmmTrainingDataTools_h
#define xmmTrainingDataTools_h

#include "xmm.h"

class XmmTrainingSetTool {
public:
	XmmTrainingSetTool() {}
	~XmmTrainingSetTool() {}

	void addPhrase(xmm::Phrase p) {
		int index;

		if (freeList.size() == 0) {
			index = set.size();
		}
		else {
			index = freeList.back();
			freeList.pop_back();
		}

		if (set.size() == 0) {
			if (p.bimodal()) {
				set = xmm::TrainingSet(xmm::MemoryMode::OwnMemory,
					xmm::Multimodality::Bimodal);
				set.dimension_input.set(p.dimension_input.get());
			}
			else {
				set = xmm::TrainingSet();
			}

			set.dimension.set(p.dimension.get());
			set.column_names.set(p.column_names, true);
		}

		set.addPhrase(index, p);
	}

	xmm::Phrase getPhrase(int index) {
		bool empty = false;

		for (int i = 0; i < freeList.size(); ++i) {
			if (index == freeList[i]) {
				empty = true;
				break;
			}
		}

		if (!empty && index < set.size() + freeList.size()) {
			return xmm::Phrase(*(set.getPhrase(index)));
		}
		else {
			//throw std::exception ?
			xmm::Phrase xp;
			return xp;
		}
	}

	void removePhrase(int index) {
		if (index >= set.size() + freeList.size()) {
			return;
		}

		for (int i = 0; i < freeList.size(); ++i) {
			if (index == freeList[i]) {
				return;
			}
		}

		freeList.push_back(index);
		std::sort(freeList.begin(), freeList.end(), std::greater<int>());
		set.removePhrase(index);
	}

	void removePhrasesOfLabel(std::string label) {
		if (!set.empty()) {
			xmm::TrainingSet *s = set.getPhrasesOfClass(label);
			if (s == nullptr) return;

			for (auto const &phrase : *s) {
				freeList.push_back(phrase.first);
			}

			std::sort(freeList.begin(), freeList.end(), std::greater<int>());
			set.removePhrasesOfClass(label);
		}
	}

	int size() {
		return static_cast<int>(set.size());
	}

	std::vector<std::string> getLabels() {
		std::set<std::string> labels = set.labels();
		std::vector<std::string> res(labels.begin(), labels.end());
		return res;
	}

	bool getBimodal() {
		return set.bimodal();
	}

	xmm::TrainingSet& getSet() {
		return set;
	}

	xmm::TrainingSet& getSubSet(std::string label) {
		return *(set.getPhrasesOfClass(label));
	}

	void setSet(xmm::TrainingSet newSet) {
		set = newSet;
	}

	void addSet(xmm::TrainingSet addSet) {
		int index;

		for (auto &xp : addSet) {
			if (freeList.size() == 0) {
				index = this->set.size();
			}
			else {
				index = freeList.back();
				freeList.pop_back();
			}

			if (set.size() == 0) {
				if (xp.second->bimodal()) {
					set = xmm::TrainingSet(xmm::MemoryMode::OwnMemory,
						xmm::Multimodality::Bimodal);
					set.dimension_input.set(xp.second->dimension_input.get());
				}
				else {
					set = xmm::TrainingSet();
				}

				set.dimension.set(xp.second->dimension.get());
				set.column_names.set(xp.second->column_names, true);
			}

			set.addPhrase(index, xp.second);
		}
	}

	void clearSet() {
		set.clear();
	}

private:
	std::vector<int> freeList;
	xmm::TrainingSet set;
};

#endif /* xmmTrainingDataTools_h */
