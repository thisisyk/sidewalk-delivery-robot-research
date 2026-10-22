"""Synthetic unit fixtures validate code only, not research performance."""
import copy
import unittest
import numpy as np
from delivery_risk.geometry import clearance
from delivery_risk.samples import Protocol,extract_features,future_labels,feature_columns


def state(x,y=0.,vx=0.,vy=0.,heading=0.):
    return dict(valid=True,x=float(x),y=float(y),z=0.,length=4.,width=2.,heading=heading,vx=float(vx),vy=float(vy))


def scene():
    return {"scenario_id":"synthetic_unit_fixture","current_time_index":10,"sdc_track_index":0,"timestamps":[i*.1 for i in range(91)],"map_features":[],"tracks":[{"id":0,"type":1,"states":[state(i*.5,vx=5.) for i in range(91)]},{"id":1,"type":1,"states":[state(20.) for i in range(91)]}]}


class GeometryTests(unittest.TestCase):
    def test_separation(self):
        self.assertAlmostEqual(clearance(state(0),state(10)),6.)
        self.assertAlmostEqual(clearance(state(0),state(0,5)),3.)
    def test_intersection_and_symmetry(self):
        a,b=state(0),state(1,heading=np.pi/4)
        self.assertEqual(clearance(a,b),0.)
        b=state(7,4,heading=.7)
        self.assertAlmostEqual(clearance(a,b),clearance(b,a))
    def test_invalid_geometry(self):
        a=state(0); a["width"]=0
        with self.assertRaises(ValueError): clearance(a,state(8))


class LeakageTests(unittest.TestCase):
    def test_future_mutation_cannot_change_inputs(self):
        a=scene(); b=copy.deepcopy(a); cfg=Protocol()
        for track in b["tracks"]:
            for s in track["states"][11:]: s.update(x=999.,y=777.,valid=False)
        b["objects_of_interest"]=["future_only"]
        b["tracks_to_predict"]=[999]
        b["sdc_paths"]=["forbidden"]
        self.assertEqual(extract_features(a,cfg),extract_features(b,cfg))
    def test_invisible_future_agent_not_selected(self):
        a=scene(); extra=copy.deepcopy(a["tracks"][1]); extra["id"]=99
        for s in extra["states"][:11]: s["valid"]=False
        a["tracks"].append(extra)
        _,ids=extract_features(a,Protocol())
        self.assertEqual(ids,[1])
    def test_labels_and_censoring(self):
        a=scene(); cfg=Protocol(); f,ids=extract_features(a,cfg)
        target=future_labels(a,ids,f,cfg)
        self.assertEqual(target["label_2m"],1)
        self.assertEqual(target["eligible_2m"],1)
        a["tracks"][1]["states"][30]["valid"]=False
        with self.assertRaisesRegex(ValueError,"future_censored"):
            future_labels(a,ids,f,cfg)
    def test_target_columns_never_inputs(self):
        cols=["ego_speed","n0_vx","neighbor_count","map_lane_distance","future_min_clearance","label_2m","eligible_2m","event_time_2m","scenario_id"]
        self.assertEqual(feature_columns(cols,"ego"),["ego_speed"])
        self.assertNotIn("map_lane_distance",feature_columns(cols,"interaction"))
        self.assertEqual(feature_columns(cols,"full"),["ego_speed","map_lane_distance","n0_vx","neighbor_count"])
    def test_current_event_is_ineligible(self):
        a=scene()
        for s in a["tracks"][1]["states"]: s["x"]=10.
        f,ids=extract_features(a,Protocol())
        self.assertEqual(future_labels(a,ids,f,Protocol())["eligible_2m"],0)


if __name__=="__main__": unittest.main()
