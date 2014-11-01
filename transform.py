import sys
import pandas as pd

def getData(file):
	dat = pd.read_csv(file, header=0)
	return dat

def getUsers1(data):
	users1 = data[['climbID', 'userID1', 'userStars1']]
	users1 = users1[pd.notnull(users1['userStars1'])]
	users1.rename(columns={'userID1':'userID', 'userStars1':'userRating'}, inplace=True)
	return users1

def getUsers2(data):
	users2 = data[['climbID', 'userID2', 'userStars2']]
	users2 = users2[pd.notnull(users2['userStars2'])]
	users2.rename(columns={'userID2':'userID', 'userStars2':'userRating'}, inplace=True)
	return users2

def getClimbs(data):
	climbs = data[['climbID', 'climbName', 'climbGrade']]
	climbs = climbs.drop_duplicates(cols='climbID')
	return climbs

def recombine(list):
	users = pd.concat(list)
	return users

if __name__ == "__main__":
	if (len(sys.argv) != 4):
		print "Usage: python transform.py inDat outDat outClimbs"
		sys.exit(1)

	# load data file
	data = getData(sys.argv[1])

	# split to users1 and users2
	u1 = getUsers1(data)
	u2 = getUsers2(data)

	# recombine
	users = recombine([u1, u2])

	# get climb data
	climbs = getClimbs(data)

	# output
	users.to_csv(sys.argv[2], header=False, index=False)
	climbs.to_csv(sys.argv[3], header=False, index=False)