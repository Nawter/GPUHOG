#!/bin/bash
# UNIX timestamp concatenated with nanoseconds
T="$(date +%s%N)"
imgDir=$(pwd)/Rendimiento/Imagenes
execDir=$(pwd)/CPU
mainDir=$(pwd)
path=../Rendimiento/Imagenes/
# Build
cd $execDir 
echo "***********************make clean****************************"
make clean
echo "***********************make ****************************"
make
cd ..
# Run
for imgDirs in "$imgDir"/*; do
	if [ -d "${imgDirs}" ] ; 
		then
		echo "**************Run*************************************"
   		echo "The folder is ::::" $imgDirs
		dirName=`basename $imgDirs`				
		for images in "$imgDirs"/*; do
		if [ -f "${images}" ] ; 	
			then				
			 	cd $execDir
			 	fileName=$(basename "$images")						 		 
			 	./OPENCV_HOG_CPU $images $fileName $dirName $execDir  
		fi
		done
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

