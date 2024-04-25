#include<iostream>
#include<stdio.h>
#include<fstream>
#include<stdlib.h>
#include<algorithm>
#include<vector>
#include<set>
#include<map>
#include<omp.h>
#include<time.h>
#include<chrono>

using namespace std;

bool cmp(const pair<int, int>& a, const pair<int, int>& b) {
    return a.second < b.second;
}

int BinarySearch(std::vector<int>& nums, int size, int t) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = left + ((right - left) / 2);
        if (nums[mid] > t) {
            right = mid - 1;
        }
        else if (nums[mid] < t) {
            left = mid + 1;
        }
        else {
            return mid;
        }
    }
    return -1;
}

void two_pointer(vector<int>& set1, int* set2, int set2_size, vector<int>& tmp_resi) {
    int p1 = 0, p2 = 0;
    while (true) {
        if (p1 >= set1.size() || p2 >= set2_size) {
            break;
        }
        if (set1[p1] > set2[p2]) {
            tmp_resi.push_back(set2[p2]);
            p2++;
        }
        else if (set1[p1] < set2[p2]) {
            p1++;
        }
        else {
            p1++;
            p2++;
        }
    }
    while (p2 < set2_size) {
        tmp_resi.push_back(set2[p2]);
        p2++;
    }
}

void cal_resi_elements(vector<int>& row_list, int id2, vector<int>& resi, vector<int>& row_id, vector<int>& col_id) {
    vector<int>tmp_resi;
    int begin_col = row_id[id2], end_col = row_id[id2 + 1];
    for (int i = begin_col; i < end_col; i++) {
        if (BinarySearch(row_list, row_list.size(), col_id[i]) == -1) {
            tmp_resi.push_back(col_id[i]);
        }
    }
    for (auto x : tmp_resi) {
        row_list.push_back(x);
    }
    sort(row_list.begin(), row_list.end());
    resi = tmp_resi;
}
 
void cal_resi_elements_2pointer(vector<int>& row_list, int id2, vector<int>& resi, vector<int>& row_id, vector<int>& col_id) {
    vector<int>tmp_resi;
    int begin_col = row_id[id2], end_col = row_id[id2 + 1];
    two_pointer(row_list, &col_id[begin_col], end_col - begin_col, tmp_resi);
    for (auto x : tmp_resi) {
        row_list.push_back(x);
    }
    sort(row_list.begin(), row_list.end());
    resi = tmp_resi;
}

struct node {
    int rows, ones, cns;
    //float profit;
    //int id;
};
    
bool compare_our(const struct node* a, const struct node* b) {
    return ((float)a->ones / a->rows) < ((float)b->ones / b->rows);
}
    
struct node struct_lst[18269000];
    
void reorder_plus(vector<int>& row_id, vector<int>& col_id, int node_num, vector<vector<int>>& res, vector<bool>& visit) {
    int block_num = (node_num + 15) / 16, windows = 300;

    vector<pair<int, int>> block_front;

    
    for (int i = 0; i < node_num; i++) {
        if (row_id[i + 1] - row_id[i] > 0) {
            auto it = col_id.begin();
            auto min_ele = min_element(it + row_id[i], it + row_id[i + 1]);
            block_front.push_back(pair<int, int>(i, *min_ele));

        }
    }
    sort(block_front.begin(), block_front.end(), cmp);
    
    vector<struct node*> node_list;
        
    for (int i = 0; i < node_num; i++) {
    //for (int i = 1; i < 3; i++) {
        struct_lst[i].ones = 0;
        struct_lst[i].rows = 0;
        struct_lst[i].cns = 0;
        //struct_lst[i].id = i;
        node_list.push_back(&struct_lst[i]);
    }
    
    int cur_ptr = 0;
    int z = 0;
    for (int z = 0; z < (node_num + 15) / 16; z++) {
    //while(true){
    //for (int z = 0; z < 1000; z++) {
        vector<int> one_block_res;
        vector<int> resi_vec;
        vector<int> all_block_rows;
        //clock_t start = clock();

        //duration += (clock() - start);
        vector<bool> is_pro(node_num);
        vector<int> is_pro_node;
        
        int flag = 0;
        while (cur_ptr < block_front.size() - 50) {
            if (!visit[block_front[cur_ptr].first]) {
                flag = 1;
                break;
            }
            cur_ptr++;
        }
        if (flag == 0) break;
        one_block_res.push_back(block_front[cur_ptr].first);
        visit[block_front[cur_ptr].first] = true;
        node_list[block_front[cur_ptr].first]->ones = 0;
        
        for (int i = row_id[block_front[cur_ptr].first]; i < row_id[block_front[cur_ptr].first + 1]; i++) {
            for (int j = row_id[col_id[i]]; j < row_id[col_id[i] + 1]; j++) {
                if (!visit[col_id[j]]) {
                    node_list[col_id[j]]->cns += 1;
                    if (!is_pro[col_id[j]]) {
                        is_pro[col_id[j]] = true;
                        is_pro_node.push_back(col_id[j]);
                    }
                }
            }
        }

        //clock_t start = clock();
        int max_id = -1;
        float max_profit = 0.0;
        int end_ptr = min(cur_ptr + windows, node_num);
//#pragma omp parallel for
        for (int i = cur_ptr; i < end_ptr; i++) {
            int act_id = block_front[i].first;
            if (!visit[act_id]) {
                node_list[act_id]->ones = (row_id[block_front[cur_ptr].first + 1] - row_id[block_front[cur_ptr].first]) + (row_id[act_id + 1] - row_id[act_id]);
                node_list[act_id]->rows = node_list[act_id]->ones - node_list[act_id]->cns;
                //node_list[i]->cns = 0;
                //node_list[act_id]->profit = (float)node_list[act_id]->ones / node_list[act_id]->rows;
                float tmp_profit = (float)node_list[act_id]->ones / node_list[act_id]->rows;
                if (tmp_profit > max_profit) {
                    max_id = act_id;
                    max_profit = tmp_profit;
                }
            }
        }
        

        if (max_id == -1) {
            res.push_back(one_block_res);
            continue;
        }
        one_block_res.push_back(max_id);
        visit[max_id] = true;
        node_list[max_id]->ones = 0;
        
        //duration += (clock() - start);

        for (int i = row_id[block_front[cur_ptr].first]; i < row_id[block_front[cur_ptr].first + 1]; i++) {
            all_block_rows.push_back(col_id[i]);
        }

        //cal_resi_elements(all_block_rows, max_id, resi_vec, row_id, col_id);
        cal_resi_elements_2pointer(all_block_rows, max_id, resi_vec, row_id, col_id);
        
        int old_ones = (row_id[block_front[cur_ptr].first + 1] - row_id[block_front[cur_ptr].first]) + (row_id[max_id + 1] - row_id[max_id]);
        int old_rows = all_block_rows.size();
        
        for (int h = 0; h < 14; h++) {
        
            int max_id = -1;
            float max_profit = 0.0;
            //vector<bool> is_pro(node_num);
            int count = 0;
            
            for (auto i : resi_vec) {
                for (int j = row_id[i]; j < row_id[i + 1]; j++) {
                    if (!visit[col_id[j]]) {
                        node_list[col_id[j]]->cns += 1;
                        if (!is_pro[col_id[j]]) {
                            is_pro[col_id[j]] = true;
                            is_pro_node.push_back(col_id[j]);
                        }
                        //count++;
                    }
                }
            }
            end_ptr = min(cur_ptr + windows, node_num);
            
//#pragma omp parallel for
            for (int i = cur_ptr; i < end_ptr; i++) {
                int act_id = block_front[i].first;
                if (!visit[act_id]) {
                    //clock_t start = clock();
                    node_list[act_id]->rows = old_rows + row_id[act_id + 1] - row_id[act_id] - node_list[act_id]->cns;
                    node_list[act_id]->ones = old_ones + row_id[act_id + 1] - row_id[act_id];
                    //node_list[i]->cns = 0;
                    //node_list[act_id]->profit = (float)node_list[act_id]->ones / node_list[act_id]->rows;
                    float tmp_profit = (float)node_list[act_id]->ones / node_list[act_id]->rows;
                    if (tmp_profit > max_profit) {
                        max_id = act_id;
                        max_profit = tmp_profit;
                    }
                    //duration += (clock() - start);
                }
            }
            
            /*for (int i = cur_ptr; i < end_ptr; i++) {
                int act_id = block_front[i].first;
                if (!visit[act_id]) {
                    if (node_list[act_id]->profit > max_profit) {
                        max_id = act_id;
                        max_profit = node_list[act_id]->profit;
                    }
                }
            }*/
            if (max_id == -1) {
                break;
            }
            one_block_res.push_back(max_id);
            visit[max_id] = true;
            node_list[max_id]->ones = 0;
            
            //cal_resi_elements(all_block_rows, max_id, resi_vec, row_id, col_id);
            cal_resi_elements_2pointer(all_block_rows, max_id, resi_vec, row_id, col_id);

            old_ones += (row_id[max_id + 1] - row_id[max_id]);
            old_rows = all_block_rows.size();
            
        }
        end_ptr = min(cur_ptr + windows, node_num);
        
//#pragma omp parallel for
        for (int i = cur_ptr; i < end_ptr; i++) {
            node_list[i]->rows = 0;
            node_list[i]->ones = 0;
            node_list[i]->cns = 0;
        }
        for (auto i : is_pro_node) {
            node_list[i]->rows = 0;
            node_list[i]->ones = 0;
            node_list[i]->cns = 0;
        }
        
        res.push_back(one_block_res);
        
    }
}

void reorder_plus_direct(vector<int>& row_id, vector<int>& col_id, int node_num, vector<vector<int>>& res, vector<bool>& visit, vector<int>& row_id_in, vector<int>& col_id_in) {
    int block_num = (node_num + 15) / 16, windows = 300;
    unsigned long long duration = 0;
    clock_t start, end;
    vector<pair<int, int>> block_front;
    //vector<bool> visit(node_num);

    for (int i = 0; i < node_num; i++) {
        if (row_id[i + 1] - row_id[i] > 0) {
            auto it = col_id.begin();
            auto min_ele = min_element(it + row_id[i], it + row_id[i + 1]);
            block_front.push_back(pair<int, int>(i, *min_ele));
            //block_front.push_back(pair<int, int>(i, 1));
        }
    }
    sort(block_front.begin(), block_front.end(), cmp);

    vector<struct node*> node_list(node_num);
    vector<bool> is_pro(node_num);

    for (int i = 0; i < node_num; i++) {
        //for (int i = 1; i < 3; i++) {
        struct_lst[i].ones = 0;
        struct_lst[i].rows = 0;
        struct_lst[i].cns = 0;
        //struct_lst[i].id = i;
        node_list[i] = &struct_lst[i];
    }

    int cur_ptr = 0;
    int z = 0;
    
    for (int z = 0; z < (node_num + 15) / 16; z++) {
        //while(true){
        //for (int z = 0; z < 1000; z++) {
         //printf("%d\n", z);
        
        vector<int> one_block_res;
        vector<int> resi_vec;
        vector<int> all_block_rows;
        //clock_t start = clock();
        //vector<bool> is_pro(node_num);
        //duration += (clock() - start);

        vector<int> is_pro_node;
        
        int flag = 0;
        while (cur_ptr < block_front.size() - 50) {
            if (!visit[block_front[cur_ptr].first]) {
                flag = 1;
                break;
            }
            cur_ptr++;
        }
        
        if (flag == 0) break;
        one_block_res.push_back(block_front[cur_ptr].first);
        visit[block_front[cur_ptr].first] = true;
        node_list[block_front[cur_ptr].first]->ones = 0;
        
        for (int i = row_id[block_front[cur_ptr].first]; i < row_id[block_front[cur_ptr].first + 1]; i++) {
            //for (int j = row_id[col_id[i]]; j < row_id[col_id[i] + 1]; j++) {
            for(int j = row_id_in[col_id[i]]; j < row_id_in[col_id[i] + 1]; j++){
                if (!visit[col_id_in[j]]) {
                    node_list[col_id_in[j]]->cns += 1;
                    if (!is_pro[col_id_in[j]]) {
                        is_pro[col_id_in[j]] = true;
                        is_pro_node.push_back(col_id_in[j]);
                    }
                }
            }
        }

        //clock_t start = clock();
        int max_id = -1;
        float max_profit = 0.0;
        int end_ptr = min(cur_ptr + windows, node_num);
        //#pragma omp parallel for
        for (int i = cur_ptr; i < end_ptr; i++) {
            int act_id = block_front[i].first;
            if (!visit[act_id]) {
                node_list[act_id]->ones = (row_id[block_front[cur_ptr].first + 1] - row_id[block_front[cur_ptr].first]) + (row_id[act_id + 1] - row_id[act_id]);
                node_list[act_id]->rows = node_list[act_id]->ones - node_list[act_id]->cns;
                //node_list[i]->cns = 0;
                //node_list[act_id]->profit = (float)node_list[act_id]->ones / node_list[act_id]->rows;
                float tmp_profit = (float)node_list[act_id]->ones / node_list[act_id]->rows;
                if (tmp_profit > max_profit) {
                    max_id = act_id;
                    max_profit = tmp_profit;
                }
            }
        }


        if (max_id == -1) {
            res.push_back(one_block_res);
            continue;
        }
        one_block_res.push_back(max_id);
        visit[max_id] = true;
        node_list[max_id]->ones = 0;

        //duration += (clock() - start);

        for (int i = row_id[block_front[cur_ptr].first]; i < row_id[block_front[cur_ptr].first + 1]; i++) {
            all_block_rows.push_back(col_id[i]);
        }

        //cal_resi_elements(all_block_rows, max_id, resi_vec, row_id, col_id);
        cal_resi_elements_2pointer(all_block_rows, max_id, resi_vec, row_id, col_id);

        int old_ones = (row_id[block_front[cur_ptr].first + 1] - row_id[block_front[cur_ptr].first]) + (row_id[max_id + 1] - row_id[max_id]);
        int old_rows = all_block_rows.size();
        
        for (int h = 0; h < 14; h++) {

            //printf("%d-%d\n", z, h);
            int max_id = -1;
            float max_profit = 0.0;
            //vector<bool> is_pro(node_num);
            int count = 0;

            for (auto i : resi_vec) {
                for (int j = row_id_in[i]; j < row_id_in[i + 1]; j++) {
                    if (!visit[col_id_in[j]]) {
                        node_list[col_id_in[j]]->cns += 1;
                        if (!is_pro[col_id_in[j]]) {
                            is_pro[col_id_in[j]] = true;
                            is_pro_node.push_back(col_id_in[j]);
                        }
                        //count++;
                    }
                }
            }
            //printf("count: %d\n", count);
            end_ptr = min(cur_ptr + windows, node_num);

            //#pragma omp parallel for
            for (int i = cur_ptr; i < end_ptr; i++) {
                int act_id = block_front[i].first;
                if (!visit[act_id]) {
                    //clock_t start = clock();
                    node_list[act_id]->rows = old_rows + row_id[act_id + 1] - row_id[act_id] - node_list[act_id]->cns;
                    node_list[act_id]->ones = old_ones + row_id[act_id + 1] - row_id[act_id];
                    //node_list[i]->cns = 0;
                    //node_list[act_id]->profit = (float)node_list[act_id]->ones / node_list[act_id]->rows;
                    float tmp_profit = (float)node_list[act_id]->ones / node_list[act_id]->rows;
                    if (tmp_profit > max_profit) {
                        max_id = act_id;
                        max_profit = tmp_profit;
                    }
                    //duration += (clock() - start);
                }
            }

            /*for (int i = cur_ptr; i < end_ptr; i++) {
                int act_id = block_front[i].first;
                if (!visit[act_id]) {
                    if (node_list[act_id]->profit > max_profit) {
                        max_id = act_id;
                        max_profit = node_list[act_id]->profit;
                    }
                }
            }*/
            if (max_id == -1) {
                break;
            }
            one_block_res.push_back(max_id);
            visit[max_id] = true;
            node_list[max_id]->ones = 0;

            //cal_resi_elements(all_block_rows, max_id, resi_vec, row_id, col_id);
            cal_resi_elements_2pointer(all_block_rows, max_id, resi_vec, row_id, col_id);

            old_ones += (row_id[max_id + 1] - row_id[max_id]);
            old_rows = all_block_rows.size();

        }
        
        end_ptr = min(cur_ptr + windows, node_num);

        //#pragma omp parallel for
        for (int i = cur_ptr; i < end_ptr; i++) {
            node_list[i]->rows = 0;
            node_list[i]->ones = 0;
            node_list[i]->cns = 0;
        }
        for (auto i : is_pro_node) {
            node_list[i]->rows = 0;
            node_list[i]->ones = 0;
            node_list[i]->cns = 0;
            is_pro[i] = false;
        }

        res.push_back(one_block_res);

    }
    
}

void readCSR(vector<int>& row_id, vector<int>& col_id, int edge_num, const char* path) {
    FILE* fp = NULL;
    fopen_s(&fp, path, "r");
    int a, b;
    int count = 1, sum = 0;
    row_id[0] = sum;
    for (int i = 0; i < edge_num; i++) {
        fscanf_s(fp, "%d,%d\n", &a, &b);
        while (b > count) {
            row_id[count++] = sum;
        }
        if (b == count) {
            col_id[sum++] = a - 1;
        }
    }
    row_id[count] = sum;
    fclose(fp);
}

void reorder_plus_new(vector<int>& row_id, vector<int>& col_id, int node_num, vector<vector<int>>& res, vector<bool>& visit) {
    int block_num = (node_num + 15) / 16;
    //unsigned long long duration = 0;
    vector<pair<int, int>> block_front;
    //vector<bool> visit(node_num);
    for (int i = 0; i < node_num; i++) {
        if (row_id[i + 1] - row_id[i] > 0) {
            auto it = col_id.begin();
            block_front.push_back(pair<int, int>(i, 1));
        }
    }
    
    vector<struct node*> node_list;
        
    for (int i = 0; i < node_num; i++) {
    //for (int i = 1; i < 3; i++) {
        struct_lst[i].ones = 0;
        struct_lst[i].rows = 0;
        struct_lst[i].cns = 0;
        //struct_lst[i].id = i;
        node_list.push_back(&struct_lst[i]);
    }
    
    int cur_ptr = 0;
    int z = 0;
    //for (int z = 0; z < (node_num + 15) / 16; z++) {
    while(true){
    //for (int z = 0; z < 1000; z++) {
        //printf("%d\n", z);
        z++;

        vector<int> one_block_res;
        vector<int> resi_vec;
        vector<int> all_block_rows;
        //clock_t start = clock();

        //duration += (clock() - start);
        vector<bool> is_pro(node_num);
        vector<int> is_pro_node;
    
        int flag = 0;
        while (cur_ptr < block_front.size()) {
            if (!visit[block_front[cur_ptr].first]) {
                flag = 1;
                break;
            }
            cur_ptr++;
        }
        if (flag == 0) break;
        one_block_res.push_back(block_front[cur_ptr].first);
        visit[block_front[cur_ptr].first] = true;
        node_list[block_front[cur_ptr].first]->ones = 0;

        for (int i = row_id[block_front[cur_ptr].first]; i < row_id[block_front[cur_ptr].first + 1]; i++) {
            for (int j = row_id[col_id[i]]; j < row_id[col_id[i] + 1]; j++) {
                if (!visit[col_id[j]]) {
                    node_list[col_id[j]]->cns += 1;
                    if (!is_pro[col_id[j]]) {
                        is_pro[col_id[j]] = true;
                        is_pro_node.push_back(col_id[j]);
                    }
                }
            }
        }

        //clock_t start = clock();
        int max_id = -1;
        float max_profit = 0.0;
        for (auto i : is_pro_node) {
            if (!visit[i]) {
                node_list[i]->ones = (row_id[block_front[cur_ptr].first + 1] - row_id[block_front[cur_ptr].first]) + (row_id[i + 1] - row_id[i]);
                node_list[i]->rows = node_list[i]->ones - node_list[i]->cns;
                //node_list[i]->cns = 0;
                float tmp_profit = (float)node_list[i]->ones / node_list[i]->rows;
                if (tmp_profit > max_profit) {
                    max_id = i;
                    max_profit = tmp_profit;
                }
            }
        }
        if (max_id == -1) {
            res.push_back(one_block_res);
            continue;
        }
        one_block_res.push_back(max_id);
        visit[max_id] = true;
        node_list[max_id]->ones = 0;
        
        //duration += (clock() - start);

        for (int i = row_id[block_front[cur_ptr].first]; i < row_id[block_front[cur_ptr].first + 1]; i++) {
            all_block_rows.push_back(col_id[i]);
        }

        cal_resi_elements(all_block_rows, max_id, resi_vec, row_id, col_id);
        
        int old_ones = (row_id[block_front[cur_ptr].first + 1] - row_id[block_front[cur_ptr].first]) + (row_id[max_id + 1] - row_id[max_id]);
        int old_rows = all_block_rows.size();

        for (int h = 0; h < 14; h++) {
        
            //printf("%d-%d\n", z, h);
            int max_id = -1;
            float max_profit = 0.0;
            //vector<bool> is_pro(node_num);
            int count = 0;
            for (auto i : resi_vec) {
                for (int j = row_id[i]; j < row_id[i + 1]; j++) {
                    if (!visit[col_id[j]]) {
                        node_list[col_id[j]]->cns += 1;
                        if (!is_pro[col_id[j]]) {
                            is_pro[col_id[j]] = true;
                            is_pro_node.push_back(col_id[j]);
                        }
                        //count++;
                    }
                }
            }
            //printf("count: %d\n", count);
            
            for (auto i : is_pro_node) {
                if (!visit[i]) {
                    //clock_t start = clock();
                    node_list[i]->rows = old_rows + row_id[i + 1] - row_id[i] - node_list[i]->cns;
                    node_list[i]->ones = old_ones + row_id[i + 1] - row_id[i];
                    //node_list[i]->cns = 0;
                    float tmp_profit = (float)node_list[i]->ones / node_list[i]->rows;
                    if (tmp_profit > max_profit) {
                        max_id = i;
                        max_profit = tmp_profit;
                    }
                    //duration += (clock() - start);
                }
            }
            if (max_id == -1) {
                break;
            }
            one_block_res.push_back(max_id);
            visit[max_id] = true;
            node_list[max_id]->ones = 0;
            
            cal_resi_elements(all_block_rows, max_id, resi_vec, row_id, col_id);

            old_ones += (row_id[max_id + 1] - row_id[max_id]);
            old_rows = all_block_rows.size();
        }
        for (auto i : is_pro_node) {
            node_list[i]->rows = 0;
            node_list[i]->ones = 0;
            node_list[i]->cns = 0;
        }
        res.push_back(one_block_res);
    }
}

void reorder_plus_new_direct(vector<int>& row_id, vector<int>& col_id, int node_num, vector<vector<int>>& res, vector<bool>& visit, vector<int>& row_id_in, vector<int>& col_id_in) {
    int block_num = (node_num + 15) / 16;
    //unsigned long long duration = 0;
    vector<pair<int, int>> block_front;
    //vector<bool> visit(node_num);
    for (int i = 0; i < node_num; i++) {
        if (row_id[i + 1] - row_id[i] > 0) {
            auto it = col_id.begin();
            block_front.push_back(pair<int, int>(i, 1));
        }
    }

    vector<struct node*> node_list;

    for (int i = 0; i < node_num; i++) {
        //for (int i = 1; i < 3; i++) {
        struct_lst[i].ones = 0;
        struct_lst[i].rows = 0;
        struct_lst[i].cns = 0;
        //struct_lst[i].id = i;
        node_list.push_back(&struct_lst[i]);
    }

    int cur_ptr = 0;
    int z = 0;
    //for (int z = 0; z < (node_num + 15) / 16; z++) {
    while (true) {
        //for (int z = 0; z < 1000; z++) {
        printf("%d\n", z);
        z++;

        vector<int> one_block_res;
        vector<int> resi_vec;
        vector<int> all_block_rows;

        vector<bool> is_pro(node_num);
        vector<int> is_pro_node;

        int flag = 0;
        while (cur_ptr < block_front.size()) {
            if (!visit[block_front[cur_ptr].first]) {
                flag = 1;
                break;
            }
            cur_ptr++;
        }
        if (flag == 0) break;
        one_block_res.push_back(block_front[cur_ptr].first);
        visit[block_front[cur_ptr].first] = true;
        node_list[block_front[cur_ptr].first]->ones = 0;
        for (int i = row_id[block_front[cur_ptr].first]; i < row_id[block_front[cur_ptr].first + 1]; i++) {
            for (int j = row_id_in[col_id[i]]; j < row_id_in[col_id[i] + 1]; j++) {
                if (!visit[col_id_in[j]]) {
                    node_list[col_id_in[j]]->cns += 1;
                    if (!is_pro[col_id_in[j]]) {
                        is_pro[col_id_in[j]] = true;
                        is_pro_node.push_back(col_id_in[j]);
                    }
                }
            }
        }


        int max_id = -1;
        float max_profit = 0.0;
        for (auto i : is_pro_node) {
            if (!visit[i]) {
                node_list[i]->ones = (row_id[block_front[cur_ptr].first + 1] - row_id[block_front[cur_ptr].first]) + (row_id[i + 1] - row_id[i]);
                node_list[i]->rows = node_list[i]->ones - node_list[i]->cns;

                float tmp_profit = (float)node_list[i]->ones / node_list[i]->rows;
                if (tmp_profit > max_profit) {
                    max_id = i;
                    max_profit = tmp_profit;
                }
            }
        }
        if (max_id == -1) {
            res.push_back(one_block_res);
            continue;
        }
        one_block_res.push_back(max_id);
        visit[max_id] = true;
        node_list[max_id]->ones = 0;


        for (int i = row_id[block_front[cur_ptr].first]; i < row_id[block_front[cur_ptr].first + 1]; i++) {
            all_block_rows.push_back(col_id[i]);
        }

        cal_resi_elements(all_block_rows, max_id, resi_vec, row_id, col_id);

        int old_ones = (row_id[block_front[cur_ptr].first + 1] - row_id[block_front[cur_ptr].first]) + (row_id[max_id + 1] - row_id[max_id]);
        int old_rows = all_block_rows.size();

        for (int h = 0; h < 14; h++) {

            int max_id = -1;
            float max_profit = 0.0;
            int count = 0;
            for (auto i : resi_vec) {
                for (int j = row_id_in[i]; j < row_id_in[i + 1]; j++) {
                    if (!visit[col_id_in[j]]) {
                        node_list[col_id_in[j]]->cns += 1;
                        if (!is_pro[col_id_in[j]]) {
                            is_pro[col_id_in[j]] = true;
                            is_pro_node.push_back(col_id_in[j]);
                        }
                    }
                }
            }

            for (auto i : is_pro_node) {
                if (!visit[i]) {
                    
                    node_list[i]->rows = old_rows + row_id[i + 1] - row_id[i] - node_list[i]->cns;
                    node_list[i]->ones = old_ones + row_id[i + 1] - row_id[i];
                    
                    float tmp_profit = (float)node_list[i]->ones / node_list[i]->rows;
                    if (tmp_profit > max_profit) {
                        max_id = i;
                        max_profit = tmp_profit;
                    }
                    
                }
            }
            if (max_id == -1) {
                break;
            }
            one_block_res.push_back(max_id);
            visit[max_id] = true;
            node_list[max_id]->ones = 0;

            cal_resi_elements(all_block_rows, max_id, resi_vec, row_id, col_id);

            old_ones += (row_id[max_id + 1] - row_id[max_id]);
            old_rows = all_block_rows.size();
        }
        for (auto i : is_pro_node) {
            node_list[i]->rows = 0;
            node_list[i]->ones = 0;
            node_list[i]->cns = 0;
        }
        res.push_back(one_block_res);
    }
}

int main() {
    //int node_num = 19717, edge_num = 88676;
    int node_num = 334925, edge_num = 1686092;
    //int node_num = 1889542, edge_num = 3946402;
    //int node_num = 43471, edge_num = 162088;
     //int node_num = 1710902, edge_num = 3636546;
    //int node_num = 410236, edge_num = 3356824;
    //int node_num = 1448038, edge_num = 5971562;
    //int node_num = 4859280, edge_num = 10149830;
    //int node_num = 3771081, edge_num = 22011034;
    //int node_num = 3138114, edge_num = 6487230;
    //int node_num = 18268992, edge_num = 172183984;

    const char* path = "DD_A.txt";
    vector<int> row_id(node_num + 1);
    vector<int> col_id(edge_num);
    vector<int> col_id_in(edge_num);
    vector<int> row_id_in(node_num + 1);
    readCSR(row_id, col_id, edge_num, path);
    for (auto x : col_id) {
        row_id_in[x + 1]++;
    }
    for (int i = 0; i < node_num; i++) {
        row_id_in[i + 1] += row_id_in[i];
    }
    vector<int> tmp_counts(node_num + 1);
    for (int i = 0; i < row_id_in.size(); i++) {
        tmp_counts[i] = row_id_in[i];
    }
    for (int i = 0; i < row_id.size() - 1; i++) {
        for (int j = row_id[i]; j < row_id[i + 1]; j++) {
            col_id_in[tmp_counts[col_id[j]]] = i;
            tmp_counts[col_id[j]]++;
        }
    }
    vector<vector<int>> res;
    vector<bool> visit(node_num);
    auto start = std::chrono::high_resolution_clock::now();
    //reorder_plus(row_id, col_id, node_num, res, visit);
    //reorder_plus_direct(row_id, col_id, node_num, res, visit, row_id_in, col_id_in);
    //reorder_plus_new(row_id, col_id, node_num, res, visit);
    reorder_plus_new_direct(row_id, col_id, node_num, res, visit, row_id_in, col_id_in);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "All time: " << duration.count() << "s" << std::endl;
    
    FILE* fp = NULL;
    fopen_s(&fp, "reorder_direct.txt", "w");

    /*for (auto x : res) {
        for (auto y : x) {
            fprintf_s(fp, "%d\n", y);
        }
    }*/
    /*for (auto x : block_front) {
        if (!visit[x.first]) {
            fprintf_s(fp, "%d\n", x.first);
            visit[x.first] = true;
        }
    }*/
    /*for (int i = 0; i < node_num; i++) {
        if (!visit[i]) {
            fprintf_s(fp, "%d\n", i);
        }
    }*/

    int count = 0;
    for (auto x : res) {
        if (x.size() == 16) {
            count++;
            for (auto y : x) {
                fprintf_s(fp, "%d\n", y);
            }
        }
    }
    for (auto x : res) {
        if (x.size() < 16) {
            for (auto y : x) {
                fprintf_s(fp, "%d\n", y);
            }
        }
    }   
    for (int i = 0; i < visit.size(); i++) {
        if (!visit[i]) fprintf_s(fp, "%d\n", i);
    }
    cout << count << endl;

    fclose(fp);
    
}
