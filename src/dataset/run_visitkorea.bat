python visitkorea_main.py --type ms --out ms_main
python visitkorea_main.py --type food --out food_main
python visitkorea_main.py --type acm --out acm_main

python visitkorea_details.py --type ms --data ms_main_list.csv --out ms_details
python visitkorea_details.py --type food --data food_main_list.csv --out food_details
python visitkorea_details.py --type acm --data acm_main_list.csv --out acm_details
