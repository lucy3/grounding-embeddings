with open("mcrae_fs.txt", "r") as mf:
    mcrae_fs = [line.strip().split("\t") for line in mf]
mcrae_fs
with open("cslb_fs.txt", "r") as cf:
    cslb_fs = [line.strip().split("\t") for line in cf]
cslb_fs
def find_overlap(f):
    overlaps = []
    for tok in f.split("_"):
        if tok in ["made", "is", "has", "a"]: continue
        overlaps.extend([(feature, cat) for feature, cat in cslb_fs if tok in feature])
    return overlaps
mcrae_fs
def find_overlap(f):
    overlaps = []
    for tok in f.split("_"):
        if tok in ["made", "is", "has", "a", "worn"]: continue
        overlaps.extend([(feature, cat) for feature, cat in cslb_fs if tok in feature])
    return overlaps
find_overlap("worn_around_neck")
def find_overlap(f):
    overlaps = []
    for tok in f.split("_"):
        if tok in ["made", "is", "has", "a", "in", "worn"]: continue
        overlaps.extend([(feature, cat) for feature, cat in cslb_fs if tok in feature])
    return set(overlaps)
all_overlaps = {f: find_overlap(f) for f in mcrae_fs.keys()}
all_overlaps = {f: find_overlap(f) for f, _ in mcrae_fs}
all_overlaps
def find_overlap(f):
    overlaps = []
    for tok in f.split("_"):
        if tok in ["made", "is", "has", "a", "in", "worn", "on", "of", "an", "for"]: continue
        overlaps.extend([(feature, cat) for feature, cat in cslb_fs if tok in feature])
    return set(overlaps)
all_overlaps
import re
def find_overlap(f):
    overlaps = []
    for tok in f.split("_"):
        if tok in ["made", "is", "has", "a", "in", "worn", "on", "of", "an", "for"]: continue
        overlaps.extend([(feature, cat) for feature, cat in cslb_fs if re.find(r"(^|_)%s(_|$)" % tok, feature)])
    return set(overlaps)
re.find?
re.match?
re.search?
def find_overlap(f):
    overlaps = []
    for tok in f.split("_"):
        if tok in ["made", "is", "has", "a", "in", "worn", "on", "of", "an", "for"]: continue
        overlaps.extend([(feature, cat) for feature, cat in cslb_fs if re.search(r"(^|_)%s(_|$)" % tok, feature)])
    return set(overlaps)
all_overlaps = {f: find_overlap(f) for f, _ in mcrae_fs}
%page all_overlaps
def find_overlap(f):
    overlaps = []
    for tok in f.split("_"):
        if tok in ["found", "the", "made", "is", "has", "a", "in", "worn", "on", "of", "an", "for"]: continue
        overlaps.extend([(feature, cat) for feature, cat in cslb_fs if re.search(r"(^|_)%s(_|$)" % tok, feature)])
    return set(overlaps)
all_overlaps = {f: find_overlap(f) for f, _ in mcrae_fs}
%page all_overlaps
def find_overlap(f):
    overlaps = []
    for tok in f.split("_"):
        if tok in ["eaten", "by", "found", "the", "made", "is", "has", "a", "in", "worn", "on", "of", "an", "for"]: continue
        overlaps.extend([(feature, cat) for feature, cat in cslb_fs if re.search(r"(^|_)%s(_|$)" % tok, feature)])
    return set(overlaps)
all_overlaps = {f: find_overlap(f) for f, _ in mcrae_fs}
%page all_overlaps
def find_overlap(f):
    overlaps = []
    for tok in f.split("_"):
        if tok in ["long", "eaten", "by", "found", "the", "made", "is", "has", "a", "in", "worn", "on", "of", "an", "for"]: continue
        overlaps.extend([(feature, cat) for feature, cat in cslb_fs if re.search(r"(^|_)%s(_|$)" % tok, feature)])
    return set(overlaps)
all_overlaps = {f: find_overlap(f) for f, _ in mcrae_fs}
%page all_overlaps
def find_overlap(f):
    overlaps = []
    for tok in f.split("_"):
        if tok in ["used", "to", "long", "eaten", "by", "found", "the", "made", "is", "has", "a", "in", "worn", "on", "of", "an", "for"]: continue
        overlaps.extend([(feature, cat) for feature, cat in cslb_fs if re.search(r"(^|_)%s(_|$)" % tok, feature)])
    return set(overlaps)
all_overlaps = {(f, cat): find_overlap(f) for f, cat in mcrae_fs}
%page all_overlaps
from pprint import ppritn
from pprint import pprint
all_overlaps
with open("overlaps", "w") as out_f:
    pprint(all_overlaps, out_f)
for f, cat in mcrae_fs:
    overlaps = find_overlap(f)
    if not overlaps: continue
    good = True
    for cand_f, cand_cat in overlaps:
        if cat in ["tactile", "smell", "sound", "taste"] and cand_cat != "other":
            good = False
        elif "visual" in cat and cand_cat != "visual":
            good = False
        elif cat == "function" and cand_cat != "functional":
            good = False
        elif cat != cand_cat:
            good = False
    if not good:
        print("%30s\t%20s" % (f, cat))
        pprint(overlaps)
        print("\n")
for f, cat in mcrae_fs:
    overlaps = find_overlap(f)
    if not overlaps: continue
    good = True
    for cand_f, cand_cat in overlaps:
        if cat in ["tactile", "smell", "sound", "taste"]:
            good = cand_cat == "other"
        elif "visual" in cat:
            good = cand_cat == "visual"
        elif cat == "function":
            good = cand_cat == "functional"
        elif cat != cand_cat:
            good = False
    if not good:
        print("%30s\t%20s" % (f, cat))
        pprint(overlaps)
        print("\n")
def find_overlap(f):
    overlaps = []
    for tok in f.split("_"):
        if tok in ["beh-", "inbeh-", "used", "to", "long", "eaten", "by", "found", "the", "made", "is", "has", "a", "in", "worn", "on", "of", "an", "for"]: continue
        overlaps.extend([(feature, cat) for feature, cat in cslb_fs if re.search(r"(^|_)%s(_|$)" % tok, feature)])
    return set(overlaps)
for f, cat in mcrae_fs:
    overlaps = find_overlap(f)
    if not overlaps: continue
    good = True
    for cand_f, cand_cat in overlaps:
        if cat in ["tactile", "smell", "sound", "taste"]:
            good = cand_cat == "other"
        elif "visual" in cat:
            good = cand_cat == "visual"
        elif cat == "function":
            good = cand_cat == "functional"
        elif cat != cand_cat:
            good = False
    if not good:
        print("%30s\t%20s" % (f, cat))
        pprint(overlaps)
        print("\n")
for f, cat in mcrae_fs:
    overlaps = find_overlap(f)
    if not overlaps: continue
    good = True
    for cand_f, cand_cat in overlaps:
        if cat in ["tactile", "smell", "sound", "taste"]:
            good = cand_cat == "other"
        elif "visual" in cat:
            good = cand_cat == "visual"
        elif cat == "function":
            good = cand_cat == "functional"
        elif cat != cand_cat:
            good = False
    if not good:
        print("%s\t%s" % (f, cat))
        pprint(overlaps)
        print("\n")
with open("bad_overlaps", "w") as bad_f:
    for f, cat in mcrae_fs:
        overlaps = find_overlap(f)
        if not overlaps: continue
        good = True
        for cand_f, cand_cat in overlaps:
            if cat in ["tactile", "smell", "sound", "taste"]:
                good = cand_cat == "other"
            elif "visual" in cat:
                good = cand_cat == "visual"
            elif cat == "function":
                good = cand_cat == "functional"
            elif cat != cand_cat:
                good = False
        if not good:
            out.write("%s\t%s\n" % (f, cat))
            pprint(overlaps, out)
            out.write("\n")
with open("bad_overlaps", "w") as bad_f:
    for f, cat in mcrae_fs:
        overlaps = find_overlap(f)
        if not overlaps: continue
        good = True
        for cand_f, cand_cat in overlaps:
            if cat in ["tactile", "smell", "sound", "taste"]:
                good = cand_cat == "other"
            elif "visual" in cat:
                good = cand_cat == "visual"
            elif cat == "function":
                good = cand_cat == "functional"
            elif cat != cand_cat:
                good = False
        if not good:
            bad_f.write("%s\t%s\n" % (f, cat))
            pprint(overlaps, bad_f)
            bad_f.write("\n")
%history
%history -f bad_overlaps.py
