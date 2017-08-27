#!/usr/bin/env python
fname="stderr.test"
dataset=["abalone", "cadata", "covetype", "e2006", "higgs", "car ins", 
		"log1p", "new20", "realsim", "susy", "year pre", "3d"]
result=["2.26841", "72842", "0.348459", "0.371375", "0.446233", "38.9214",
		"0.37719", "0.784022", "0.705245", "0.376718", "9.34178", "3.42847e+07"]
new_result=[]

with open(fname) as f:
	    content = f.readlines()
content = [x.strip() for x in content] 
for line in content:
	parts = line.split("=")
	if parts[0]=="rmse":
		new_result.append(parts[1])
for i in range(0, len(result)):
	if new_result[i] != result[i]:
		print "diff dataset %s: %s v.s. %s" % (dataset[i], new_result[i], result[i])

print "auto testing completed!"
