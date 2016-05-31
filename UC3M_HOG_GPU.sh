#!/bin/bash
# UNIX timestamp concatenated with nanoseconds
T="$(date +%s%N)" 
imgDir=$(pwd)/Rendimiento/Imagenes
execDir=$(pwd)/GPU/UC3M_HOG_GPU
mainDir=$(pwd)
# Remove files 
cd $execDir
if [ -a Makefile ] ;
	then
	echo "*****Inside make disclean****************"
	make distclean	
fi
echo "***********Outside make distclean*****************"
rm -rf ../lib/lib*
rm -rf ../bin/*
rm -rf ../tmp/*
# Build 
/usr/bin/qmake 
make all
cd UC3M_HOG_GPU
mv lib* ../lib/
cd ..
# Run
echo $(pwd)
echo $imgDir
# find . -name "*.sift"  -exec rm -rf {} \;
for imgDirs in "$imgDir"/*; do
# # distinguir entre dirs y files
     if [ -d "${imgDirs}" ] ; 
 		then 
 		    echo "**************Run*************************************"
   		    echo "The folder is ::::" $imgDirs 
 			cd $execDir
 			./bin/Main tmp/ $imgDirs model/config
 			# Time interval in nanoseconds
			T="$(($(date +%s%N)-T))"
			# Seconds
			S="$((T/1000000000))"
			# Milliseconds
			M="$((T/1000000))"
			printf "The total time is : %02d:%02d:%02d:%02d.%03d\n" "$((S/86400))" "$((S/3600%24))" "$((S/60%60))" "$((S%60))" "${M}"
			T="$(date +%s%N)"		
             cd $mainDir         
 	fi	
 done
