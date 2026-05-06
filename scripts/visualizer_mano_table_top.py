# -*- coding: utf-8 -*-
import sys

sys.path = ["."] + sys.path

from scripts.visualizer_sharpa_table_top import main


if __name__ == "__main__":
    main(
        script_name="visualizer_mano_table_top.py",
        hand_model_filter="mano",
        collection_example="mano_tabletop",
    )
