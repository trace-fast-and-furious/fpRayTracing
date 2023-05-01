#ifndef HITTABLE_LIST_BFP_H
#define HITTABLE_LIST_BFP_H

#include "hittable.cuh"

using namespace custom_precision_fp;

class hittable_list : public hittable
{
public:
    __device__ __host__ hittable_list() {}
    __device__ __host__ hittable_list(hittable **l, int n)
    {
        list = l;
        list_size = n;
    }
    __device__ __host__ virtual bool hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const;
    __device__ __host__ virtual bool hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const;

    hittable **list;
    int list_size;
};

bool hittable_list::hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const
{
    fp_custom t_min = fp_orig_to_custom(__t_min);
    fp_custom t_max = fp_orig_to_custom(__t_max);

    hit_record temp_rec;
    bool hit_anything = false;
    fp_custom closest_so_far = t_max;
    for (int i = 0; i < list_size; i++)
    {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

bool hittable_list::hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    fp_custom closest_so_far = t_max;
    for (int i = 0; i < list_size; i++)
    {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

#endif