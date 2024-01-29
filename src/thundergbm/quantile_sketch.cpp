//
// Created by qinbin on 2018/5/9.
//

#include "thundergbm/quantile_sketch.h"
#include <math.h>

void summary::Reserve(int size){
    if(size > entries.size()){
        entry_reserve_size = size;
        entries.resize(size);
        //data...
    }
}

void summary::Copy(summary& src){
    entry_size = src.entry_size;
    entry_reserve_size = src.entry_reserve_size;
    entries = src.entries;
}

void summary::Merge(summary& src1, summary& src2){
    if(src1.entry_size == 0 && src2.entry_size == 0){
        this->entry_size = 0;
        this->entry_reserve_size = 0;
        this->entries.clear();
        return;
    }
    else if(src1.entry_size == 0){
        this->Copy(src2);
        return;
    }
    else if(src2.entry_size == 0){
        this->Copy(src1);
        return;
    }
    float_type r1 = 0;
    float_type r2 = 0;
    int i = 0, j = 0;
    this->Reserve(src1.entry_size + src2.entry_size);
    this->entry_size = 0;
    for(; i < src1.entry_size && j < src2.entry_size;){
        int val1 = src1.entries[i].val;
        int val2 = src2.entries[j].val;
        if(val1 == val2){
            CHECK(this->entry_size < entry_reserve_size) << this->entry_size;
            this->entries[this->entry_size++] = entry(val1,
                                     src1.entries[i].rmin + src2.entries[j].rmin,
                                     src1.entries[i].rmax + src2.entries[j].rmax,
                                     src1.entries[i].w + src2.entries[j].w);
            r1 = src1.entries[i].rmin + src1.entries[i].w;
            r2 = src2.entries[j].rmin + src2.entries[j].w;
            i++;
            j++;
			//this->entry_size++;
        }
        else if(val1 < val2){
            CHECK(this->entry_size < entry_reserve_size) << this->entry_size;
            this->entries[this->entry_size++]=entry(val1,
                                     src1.entries[i].rmin + r2,
                                     src1.entries[i].rmax + src2.entries[j].rmax - src2.entries[j].w,
                                          src1.entries[i].w);
            r1 = src1.entries[i].rmin + src1.entries[i].w;
            i++;
			//this->entry_size++;
        }
        else{
            CHECK(this->entry_size < entry_reserve_size) << this->entry_size;
            this->entries[this->entry_size++] = entry(val2,
                                     src2.entries[j].rmin + r1,
                                     src2.entries[j].rmax + src1.entries[i].rmax - src1.entries[i].w,
                                     src2.entries[j].w);
            r2 = src2.entries[j].rmin + src2.entries[j].w;
            j++;
			//this->entry_size++;
        }
    }
    for(; i < src1.entry_size; i++){
        CHECK(this->entry_size < entry_reserve_size) << this->entry_size;
        this->entries[this->entry_size++] = entry(src1.entries[i].val,
                                 src1.entries[i].rmin + r2,
                                 src1.entries[i].rmax + src2.entries[src2.entry_size - 1].rmax,
                                 src1.entries[i].w);
    	//this->entry_size++;
	}
    for(; j < src2.entry_size; j++){
        CHECK(this->entry_size < entry_reserve_size) << this->entry_size;
        this->entries[this->entry_size++] = entry(src2.entries[j].val,
                                 src2.entries[j].rmin + r1,
                                 src2.entries[j].rmax +  src1.entries[src1.entry_size - 1].rmax,
                                 src2.entries[j].w);
    	//this->entry_size++;
	}
    //this->entry_size = this->entries.size();
    r1 = 0;
    r2 = 0;
//    float_type rmin_diff = 0;
//    float_type rmax_diff = 0;
//    float_type w_diff = 0;
    for(int i = 0; i < this->entry_size; i++){
        if(this->entries[i].rmin < r1){
            this->entries[i].rmin = r1;
//            if(r1 - this->entries[i].rmin > rmin_diff)
//                rmin_diff = r1 - this->entries[i].rmin;
        }
		else
        	r1 = this->entries[i].rmin;
        if(this->entries[i].rmax < r2){
            this->entries[i].rmax = r2;
//            if(r2 - this->entries[i].rmax > rmax_diff)
//                rmax_diff = r2 - this->entries[i].rmax;
        }
        if(this->entries[i].rmax < this->entries[i].rmin + this->entries[i].w){
            this->entries[i].rmax = this->entries[i].rmin + this->entries[i].w;
//            if(this->entries[i].rmax - this->entries[i].rmin - this->entries[i].w > w_diff)
//                    w_diff = this->entries[i].rmax - this->entries[i].rmin - this->entries[i].w;
        }
		r2 = this->entries[i].rmax;
    }
}



void summary::Prune(summary& src, int size){
    if(src.entry_size <= size){
        this->Copy(src);
        return;
    }
    float_type begin = src.entries[0].rmax;
    float_type End = src.entries[src.entry_size - 1].rmin;
    float_type range = End - begin;
    if(size <= 2 || range == 0.0f){
        this->entry_size = 2;
        CHECK(1 < entry_reserve_size) << entry_reserve_size;
        this->entries[0] = src.entries[0];
        this->entries[1] = src.entries[src.entry_size - 1];
        return;
    }
    range = (std::max)(range, (float_type)1e-3f);
    int n_points = size - 2;
    int n_bigbin = 0;
    int safe_factor = 2;
    float_type chunk_size = safe_factor * range / n_points;
    float_type sum_small_range = 0;
    int j = 0;
    int i = 1;
    float_type r1;
    float_type r2;
    vector<int> big_points;
    big_points.reserve(n_points);
    //int last_big_point = 0;
    for(; i < src.entry_size - 1; i++){
        CHECK(i < src.entry_reserve_size) << i;
        r1 = src.entries[i].rmin + src.entries[i].w;
        r2 = src.entries[i].rmax - src.entries[i].w;
        if(r1 > r2 + chunk_size){
            n_bigbin++;
            big_points.push_back(i);
            //
            if(j != i -1)
                sum_small_range += r2 - (src.entries[j].rmin + src.entries[j].w);
            j = i;
        }
    }
    CHECK(n_bigbin < n_points) << "too many big bin";
    int n_smallbin = n_points - n_bigbin;
    //r1 = src.entries[i].rmin + src.entries[i].w;
    r2 = src.entries[i].rmax - src.entries[i].w;
    if(j != src.entry_size - 2)
        sum_small_range += r2 - (src.entries[j].rmin + src.entries[j].w);
    CHECK(j < src.entry_reserve_size) << j;
    this->entries[0] = src.entries[0];
    this->entry_size = 1;
	n_points -= n_bigbin;
    j = 0;
    int n_get_points = 1;
    //store maximum point
    big_points.push_back(src.entry_size - 1);
    for(int i = 0; i < big_points.size(); i++){
        int id = big_points[i];
        if(j != id -1){
            CHECK(id < src.entry_reserve_size) << id;
            float_type r = src.entries[id].rmax - src.entries[id].w;
            int k = j;
            for(; n_get_points < n_points; n_get_points++){
                float_type start = n_get_points * sum_small_range / n_points + begin;
                if(start >= r)
                    break;
                for(; k < id; k++){
                    CHECK(k+1 < src.entry_reserve_size) << k+1;
                    if(2 * start < (src.entries[k + 1].rmax + src.entries[k + 1].rmin))
                        break;
                }
                if(k == id) break;
                CHECK(k < src.entry_reserve_size) << k;
                if(2 * start >= src.entries[k].rmin + src.entries[k].w + src.entries[k+1].rmax - src.entries[k+1].w){
                    if(k != j - 1){
                        j = k + 1;
                        CHECK(k < src.entry_reserve_size - 1) << k;
                        this->entries[this->entry_size] = src.entries[k + 1];
                        this->entry_size++;
						
                    }
                }
                else{
                    if(k != j){
                        j = k;
                        CHECK(k < src.entry_reserve_size) << k;
                        this->entries[this->entry_size] = src.entries[k];
                        this->entry_size++;
                    }
                }
            }
        }
        //store big bin
        if(j != id){
            CHECK(id < src.entry_reserve_size) << id;
            this->entries[this->entry_size] = src.entries[id];
            this->entry_size++;
            j = id;
        }
        CHECK(j < src.entry_reserve_size) << j;
        begin += src.entries[j].rmin + 2 * src.entries[j].w - src.entries[j].rmax;
    }
//    for(int i = 1; i < src.entry_size; i++){
//        if
//    }


};

void Qitem::GetSummary(summary& ret){
    //remove it if data is sorted
    //sort(data.begin(), data.begin() + tail);
    ret.entry_size = 0;
    float_type waccum = 0;
    for(int i = 0; i < tail;){
        int j = i + 1;
        CHECK(i < data.size()) << i;
        float_type wt = data[i].second;
        for(; j < tail; j++){
            CHECK(j < data.size()) << j;
            if(data[j].first == data[i].first)
                wt += data[j].second;
            else
                break;
        }
        CHECK(ret.entry_size < ret.entry_reserve_size) << ret.entry_size;
        ret.entries[ret.entry_size] = entry(data[i].first, waccum, waccum + wt, wt);
        ret.entry_size++;
        waccum += wt;
        i = j;
    }
}

void quanSketch::Init(int maxn, float_type eps){
    numOfLevel = 1;
    while (1) {
        summarySize = ceil(numOfLevel / eps) + 1;
        int n = (1ULL << numOfLevel);
        //break when summary size is big enough (one cut point candidate has one or zero instances on average.)
        if (n * summarySize >= maxn) break;
        ++numOfLevel;
    }
//    std::cout<<"summarySize:"<<summarySize<<std::endl;
	int n = (1ULL << numOfLevel);
    CHECK(n * summarySize >= maxn) << "invalid init parameter";
    CHECK(numOfLevel <= summarySize * eps) << "invalid init parameter";
    Qentry.data.clear();
    Qentry.data.resize(summarySize * 2);
    Qentry.tail = 0;
    //summaries.clear();
}


void quanSketch::Add(float_type value, float_type weight){
    if(weight == 0.0f) return;
    if(Qentry.data.size() == Qentry.tail){
        t_summary.Reserve(2*summarySize);
        Qentry.GetSummary(t_summary);
        Qentry.tail = 0;
        for(int i = 1;; i++){
            if(summaries.size() < i + 1){
				//Qentry.data.resize((i+1)*summarySize);
                summaries.resize(i + 1, summary(0, (i+1) * summarySize));
            }
            CHECK(i < summaries.size()) << i;
            if(summaries[i].entry_size == 0){
                summaries[i].Prune(t_summary, summarySize);
                break;
            }
            else{
                summaries[0].Prune(t_summary, summarySize);
                CHECK(i < summaries.size()) << i;
                t_summary.Merge(summaries[0], summaries[i]);
                if(t_summary.entry_size > summarySize)
                    summaries[i].entry_size = 0;
                else{
                    summaries[i].Copy(t_summary);
                    break;
                }
            }
        }
//        this->AddT();
    }
    CHECK(Qentry.tail < Qentry.data.size()) << Qentry.tail;
    if(Qentry.tail == 0 || value != Qentry.data[Qentry.tail-1].first){
        CHECK(Qentry.tail < Qentry.data.size()) << Qentry.tail;
		Qentry.data[Qentry.tail] = std::make_pair(value, weight);
		Qentry.tail++;
	}
	else{
        CHECK(Qentry.tail <= Qentry.data.size()) << Qentry.tail;
        Qentry.data[Qentry.tail-1].second += weight;
    }

	//Qentry.data.push_back(std::make_pair(value, weight));
}

void quanSketch::GetSummary(summary& dest){
    dest.entry_size = 0;
    dest.entries.clear();
    if(summaries.size() == 0){
//		std::cout<<"0 size"<<std::endl;
        dest.Reserve(Qentry.data.size());
        Qentry.GetSummary(dest);
        if(dest.entry_size > summarySize){
            t_summary.Reserve(summarySize);
            t_summary.Prune(dest, summarySize);
            dest.Copy(t_summary);
        }
    }
    else {
//		std::cout<<"not 0 size"<<std::endl;
        dest.Reserve(2 * summarySize);
        Qentry.GetSummary(dest);
        summaries[0].Prune(dest, summarySize);
        for(int i = 1; i < summaries.size(); i++){
            if(summaries[i].entry_size == 0)
                continue;
			if(summaries[0].entry_size == 0)
				summaries[0].Copy(summaries[i]);
			else{
            	dest.Merge(summaries[0], summaries[i]);
            	summaries[0].Prune(dest, summarySize);
			}
        }
        dest.Copy(summaries[0]);
    }

}

//void quanSketch::AddT(){
//    t_summary
//}
