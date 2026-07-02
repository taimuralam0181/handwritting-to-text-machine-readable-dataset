Seed dataset for offline department classification.

Purpose:
- Binary medicine department labels for `Cardiology` and `Ophthalmology`
- Intended as a starter dataset for local classifier training
- Curated from publicly accessible WHO / national essential medicines list sources

Primary sources used:
- WHO national essential medicines list snippet with cardiovascular medicines:
  https://cdn.who.int/media/docs/default-source/essential-medicines/national-essential-medicines-lists-%28neml%29/afro_neml/seychelles-2022.pdf?sfvrsn=f13f1b70_3
- WHO national essential medicines list snippet with ophthalmological preparations:
  https://cdn.who.int/media/docs/default-source/essential-medicines/national-essential-medicines-lists-%28neml%29/wpro_neml/malaysia-national-essential-medicines-list-7th-v04.pdf?download=true&sfvrsn=e1bbca5d_1
- WHO national essential medicines list snippet with additional ophthalmic examples:
  https://cdn.who.int/media/docs/default-source/essential-medicines/national-essential-medicines-lists-%28neml%29/emro_neml/somalia-2019.pdf?download=true&sfvrsn=3d4b3f4f_3
- WHO national essential medicines list snippet with additional cardiovascular examples:
  https://cdn.who.int/media/docs/default-source/essential-medicines/national-essential-medicines-lists-%28neml%29/afro_neml/uganda-2023.pdf?download=true&sfvrsn=5205bf98_3

Notes:
- This is a curated seed dataset, not a complete benchmark dataset.
- Some rows include dosage-form wording like `eye drops`, `ophthalmic solution`, or `intravitreal injection` because those phrases help distinguish eye prescriptions from cardiology prescriptions.
- For a production-grade classifier, expand this file with local brand names used in your prescriptions.
