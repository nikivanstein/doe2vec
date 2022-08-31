import json
  
# Opening JSON file
f = open('f1_results.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)
print(data)
f.close()
print( "\\toprule")
print( "Task \t&\t", end="")

for alg in data:
    print( alg + "\t&\t", end="")
print("\\\\")
print("\\midrule")

for prob in data["AE16"]:
    print(prob + "\t&\t", end="")
    for alg in data:
        print(format(data[alg][prob],".3f"), end="")
        print("\t&\t", end="")
    print("\\\\")