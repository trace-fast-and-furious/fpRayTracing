#ifndef HITTABLE_LIST_BFP_H
#define HITTABLE_LIST_BFP_H

#include "hittable.h"

using namespace floating_point;

class hittable_list : public hittable
{
public:
    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }
    void add(shared_ptr<hittable> object) { objects.push_back(object); }

    virtual bool hit(const ray &r, fp t_min, fp t_max, hit_record &rec) const override;
    virtual bool hit(const ray &r, __fpo __t_min, __fpo __t_max, hit_record &rec) const override;
    virtual bool bounding_box(fp time0, fp time1, aabb &output_box) const override;
    virtual bool bounding_box(__fpo __time0, __fpo __time1, aabb &output_box) const override;

public:
    std::vector<shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const ray &r, fp t_min, fp t_max, hit_record &rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    fp closest_so_far = t_max;

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

bool hittable_list::hit(const ray &r, __fpo __t_min, __fpo __t_max, hit_record &rec) const
{
    fp t_min = fpo_to_fp(__t_min);
    fp t_max = fpo_to_fp(__t_max);

    hit_record temp_rec;
    bool hit_anything = false;
    fp closest_so_far = t_max;

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

bool hittable_list::bounding_box(fp time0, fp time1, aabb &output_box) const
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

bool hittable_list::bounding_box(__fpo __time0, __fpo __time1, aabb &output_box) const
{
    fp time0 = fpo_to_fp(__time0);
    fp time1 = fpo_to_fp(__time1);
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