"""人工构造的微型数学样例；不进入正式 P2 统计。"""
import csv
import gzip
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from q1.conflict.analysis import Summary, domain_rows, wilson_interval
from q1.conflict.clustering import adjusted_rand, cluster_labels, distance_matrix, silhouette_from_distance
from q1.conflict.input import iter_p1_batches
from q1.conflict.reporting import aggregate_dimension_contributions, write_tables
from q1.conflict.scoring import prepare_pairs, score_batch


class ConflictTests(unittest.TestCase):
    def setUp(self):
        self.weights=np.array([.2,.3,.5])
        self.groups=[{"dimension":"G1","indices":np.array([0,1]),"indicator_weights":np.array([.4,.6])},
                     {"dimension":"G2","indices":np.array([2]),"indicator_weights":np.array([1.])}]
        self.pairs=prepare_pairs(self.weights)

    def test_hand_calculated_conflict_and_pair_decomposition(self):
        z=np.array([[.5,.5,.5],[0,0,1],[1,0,1]],dtype=float)
        scored=score_batch(z,self.groups,self.pairs)
        self.assertAlmostEqual(scored["conflict"][0],0)
        denominator=.2*.3+.2*.5+.3*.5
        self.assertAlmostEqual(scored["conflict"][1],(.2*.5+.3*.5)/denominator)
        self.assertTrue(np.allclose(scored["conflict"],scored["pair_contributions"].sum(axis=1)))
        self.assertTrue(np.all((scored["conflict"]>=0)&(scored["conflict"]<=1)))
        self.assertTrue(np.all((scored["dimension"]>=0)&(scored["dimension"]<=1)))
        self.assertAlmostEqual(scored["dimension"][2,0],.4)
        self.assertAlmostEqual(scored["dimension"][2,1],1.)

    def test_zero_denominator_rejected(self):
        with self.assertRaises(ValueError):prepare_pairs(np.array([1.,0.,0.]))

    def test_fixed_threshold_domain_rate_and_contribution(self):
        reference=np.array([0.,.2,.4,.6,.8,1.])
        thresholds=np.quantile(reference,[.85,.90,.95])
        self.assertTrue(np.all(np.diff(thresholds)>=0))
        z=np.array([[0,0,1],[.5,.5,.5]],dtype=float)
        score=score_batch(z,self.groups,self.pairs)
        summary=Summary(3,1,thresholds)
        summary.add("A2","arxiv",np.array([70.,40.]),score)
        bucket=summary.buckets[("A2","arxiv")]
        expected=int(np.count_nonzero(score["conflict"]>thresholds[1]))
        self.assertEqual(bucket.high_n[1],expected)
        self.assertAlmostEqual(bucket.high_n[1]/bucket.n,expected/2)
        self.assertTrue(np.allclose(bucket.pair_sum[1].sum(),score["conflict"][score["conflict"]>thresholds[1]].sum()))
        rows=domain_rows(summary,[.85,.90,.95],200,7)
        row=next(r for r in rows if r["domain"]=="arxiv" and r["quantile"]==.9)
        self.assertEqual(row["high_n"],expected)
        self.assertAlmostEqual(row["conflict_rate"],expected/2)
        self.assertEqual(wilson_interval(1419,0)[0],0)
        self.assertGreater(wilson_interval(1419,0)[1],0)

    def test_dimension_weighted_contribution_sums_to_conflict(self):
        z=np.array([[0.,0.,1.]],dtype=float)
        score=score_batch(z,self.groups,self.pairs)
        first,second,_,_=self.pairs
        grouped=aggregate_dimension_contributions(score["pair_contributions"][0],first,second,{0:"G1",1:"G1",2:"G2"})
        self.assertAlmostEqual(sum(grouped.values()),score["conflict"][0])
        self.assertIn(("G1","G2"),grouped)

    def test_written_dimension_contributions_reconstruct_mean_conflict(self):
        thresholds=np.array([.1,.2,.3])
        z=np.array([[0.,0.,1.],[1.,0.,1.]])
        score=score_batch(z,self.groups,self.pairs)
        summary=Summary(3,1,thresholds)
        summary.add("union","arxiv",np.array([40.,60.]),score)
        groups=[{"dimension":"G1","indices":np.array([0,1]),"label":"group 1","weight_sum":.5},
                {"dimension":"G2","indices":np.array([2]),"label":"group 2","weight_sum":.5}]
        with tempfile.TemporaryDirectory() as temp:
            result=write_tables(Path(temp),["a","b","c"],groups,self.pairs,(np.array([0]),np.array([1])),summary,
                                [.85,.9,.95],[{"view":"union"}],[{"view":"union"}],[{"view":"union"}],[{"view":"union"}])
            weighted=result[2]
            selected=[r for r in weighted if r["view"]=="union" and r["domain"]=="all" and r["quantile"]==.9]
            mask=score["conflict"]>thresholds[1]
            self.assertAlmostEqual(sum(r["mean_weighted_contribution"] for r in selected),float(np.mean(score["conflict"][mask])))

    def test_correlation_distance_signed_and_clustering(self):
        corr=np.array([[1.,.9,-.9,0.],[.9,1.,-.8,0.],[-.9,-.8,1.,.8],[0.,0.,.8,1.]])
        dist=distance_matrix(corr)
        self.assertAlmostEqual(dist[0,2],1.9)
        self.assertAlmostEqual(distance_matrix(corr,absolute=True)[0,2],.1)
        tree,labels=cluster_labels(dist,"average",2)
        self.assertEqual(len(set(labels)),2)
        self.assertEqual(adjusted_rand(labels,labels),1.)
        self.assertTrue(-1<=silhouette_from_distance(dist,labels)<=1)
        with self.assertRaises(ValueError):cluster_labels(dist,"ward",2)
        broken=corr.copy();broken[0,1]=0
        with self.assertRaises(ValueError):distance_matrix(broken)

    def test_p1_input_version_and_q_check(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"micro.csv.gz"
            fields=["record_id","source_file","line_number","record_hash","attachment","domain","audit_status",
                    "model_version","union_membership","Q","z_a","z_b","z_c"]
            row={"record_id":"x","source_file":"fixture","line_number":"1","record_hash":"h","attachment":"A2",
                 "domain":"arxiv","audit_status":"valid","model_version":"q1-p1-v1","union_membership":"True",
                 "Q":"50","z_a":".5","z_b":".5","z_c":".5"}
            with gzip.open(path,"wt",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerow(row)
            batches=list(iter_p1_batches(path,["a","b","c"],self.weights,"q1-p1-v1",2,{"A2"}))
            self.assertEqual(len(batches[0][0]),1)
            with self.assertRaises(RuntimeError):list(iter_p1_batches(path,["a","b","c"],self.weights,"wrong",2,{"A2"}))


if __name__=="__main__":unittest.main()
