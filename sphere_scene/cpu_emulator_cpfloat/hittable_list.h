#ifndef HITTABLE_LIST_BFP_H
#define HITTABLE_LIST_BFP_H

#include "hittable.h"

using namespace custom_precision_fp;

class hittable_list : public hittable
{
public:
    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }
    void add(shared_ptr<hittable> object) { objects.push_back(object); }

    virtual bool hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const override;
    virtual bool hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const override;
    virtual bool bounding_box(fp_custom time0, fp_custom time1, aabb &output_box) const override;
    virtual bool bounding_box(fp_orig __time0, fp_orig __time1, aabb &output_box) const override;

public:
    std::vector<shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    fp_custom closest_so_far = t_max;

    for (const auto &object : objects)
    {
        if (object->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

bool hittable_list::hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const
{
    fp_custom t_min = fp_orig_to_custom(__t_min);
    fp_custom t_max = fp_orig_to_custom(__t_max);

    hit_record temp_rec;
    bool hit_anything = false;
    fp_custom closest_so_far = t_max;

    for (const auto &object : objects)
    {
        if (object->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

bool hittable_list::bounding_box(fp_custom time0, fp_custom time1, aabb &output_box) const
{
    if (objects.empty())
        return false;
    aabb temp_box;
    bool first_box = true;

    for (const auto &object : objects)
    {
        if (!object->bounding_box(time0, time1, temp_box)) // 함수가 제대로 실행되지 않을 경우
            return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }
    return true;
}

bool hittable_list::bounding_box(fp_orig __time0, fp_orig __time1, aabb &output_box) const
{
    fp_custom time0 = fp_orig_to_custom(__time0);
    fp_custom time1 = fp_orig_to_custom(__time1);
    if (objects.empty())
        return false;
    aabb temp_box;
    bool first_box = true;

    for (const auto &object : objects)
    {
        if (!object->bounding_box(time0, time1, temp_box)) // 함수가 제대로 실행되지 않을 경우
            return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }
    return true;
}

#endif