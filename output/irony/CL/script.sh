mkdir hinge/ log/
for d in $(find . -type d -links 2)
do
   mkdir "${d}"/MAX/ "${d}"/MAJORITY/

done

for d in $(find . -type d -links 2)
do
 for i in {1..10}
   do
   echo "${d}"
   mkdir  "${d}"/trails_$((i))/
 done
done

for d in $(ls)
do 
	fp

