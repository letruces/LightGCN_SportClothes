#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <time.h>

typedef unsigned int ui;

using namespace std;
namespace py = pybind11;

int randint_(int end)
{
    if (end <= 0) return 0; // Tối ưu: Tránh chia cho 0
    return rand() % end;
}

py::array_t<int> sample_negative(int user_num, int item_num, int train_num, const std::vector<std::vector<int>>& allPos, int neg_num)
{
    int perUserNum = (user_num > 0) ? (train_num / user_num) : 0;
    if (perUserNum == 0) perUserNum = 1;

    int row = neg_num + 2;
    py::array_t<int> S_array = py::array_t<int>({user_num * perUserNum, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user = 0; user < user_num; user++)
    {
        const std::vector<int>& pos_item = allPos[user];
        
        // KIỂM TRA TÍNH HỢP LỆ: Nếu user không có item nào thì bỏ qua để không crash
        if (pos_item.empty()) {
            for (int pair_i = 0; pair_i < perUserNum; pair_i++) {
                int base_idx = (user * perUserNum + pair_i) * row;
                ptr[base_idx] = user;
                ptr[base_idx + 1] = 0;
                for (int index = 2; index < row; index++) {
                    ptr[base_idx + index] = randint_(item_num);
                }
            }
            continue;
        }

        // Tối ưu dùng unordered_set để tìm kiếm O(1) thay vì std::find O(N)
        std::unordered_set<int> pos_set(pos_item.begin(), pos_item.end());

        for (int pair_i = 0; pair_i < perUserNum; pair_i++)
        {
            int base_idx = (user * perUserNum + pair_i) * row;
            ptr[base_idx] = user;
            ptr[base_idx + 1] = pos_item[randint_(pos_item.size())];
            
            for (int index = 2; index < neg_num + 2; index++)
            {
                int negitem = 0;
                int count = 0;
                do
                {
                    negitem = randint_(item_num);
                    count++;
                    if (count > 100) break; // Tránh lặp vô tận
                } while (pos_set.find(negitem) != pos_set.end());
                
                ptr[base_idx + index] = negitem;
            }
        }
    }
    return S_array;
}

py::array_t<int> sample_negative_ByUser(const std::vector<int>& users, int item_num, const std::vector<std::vector<int>>& allPos, int neg_num)
{
    int row = neg_num + 2;
    int col = users.size();
    py::array_t<int> S_array = py::array_t<int>({col, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user_i = 0; user_i < col; user_i++)
    {
        int user = users[user_i];
        const std::vector<int>& pos_item = allPos[user];
        int base_idx = user_i * row;

        ptr[base_idx] = user;

        if (pos_item.empty()) {
            ptr[base_idx + 1] = 0;
            for (int neg_i = 2; neg_i < row; neg_i++) {
                ptr[base_idx + neg_i] = randint_(item_num);
            }
            continue;
        }

        std::unordered_set<int> pos_set(pos_item.begin(), pos_item.end());
        ptr[base_idx + 1] = pos_item[randint_(pos_item.size())];

        for (int neg_i = 2; neg_i < row; neg_i++)
        {
            int negitem = 0;
            int count = 0;
            do
            {
                negitem = randint_(item_num);
                count++;
                if (count > 100) break;
            } while (pos_set.find(negitem) != pos_set.end());
            
            ptr[base_idx + neg_i] = negitem;
        }
    }
    return S_array;
}

void set_seed(unsigned int seed)
{
    srand(seed);
}

using namespace py::literals;

PYBIND11_MODULE(sampling, m)
{
    srand(time(0));
    m.doc() = "sampling extension for LightGCN";
    m.def("randint", &randint_, "generate int between [0 end]", "end"_a);
    m.def("seed", &set_seed, "set random seed", "seed"_a);
    m.def("sample_negative", &sample_negative, "sampling negatives for all",
          "user_num"_a, "item_num"_a, "train_num"_a, "allPos"_a, "neg_num"_a);
    m.def("sample_negative_ByUser", &sample_negative_ByUser, "sampling negatives for given users",
          "users"_a, "item_num"_a, "allPos"_a, "neg_num"_a);
}