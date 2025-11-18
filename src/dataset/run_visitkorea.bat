python test.py --type ms --out ms_main --empty-patience 2
python test.py --type food --out food_main --empty-patience 2
python test.py --type acm --out acm_main --empty-patience 2

python visitkorea_details.py --type ms --data ms_main_list.csv --out ms_details --limit 3
python visitkorea_details.py --type food --data food_main_list.csv --out food_details --limit 3
python visitkorea_details.py --type acm --data acm_main_list.csv --out acm_details --limit 3
