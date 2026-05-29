import importlib

from src.archetype_labels import build_archetype_name_summary


playoff_projection = importlib.import_module("pipelines.09_project_playoff_archetypes")


def test_regular_feature_to_playoff_feature_supports_moneypuck_prefixes():
    assert playoff_projection.regular_feature_to_playoff_feature("reg_points_per60") == "po_points_per60"
    assert (
        playoff_projection.regular_feature_to_playoff_feature("mp_reg_5on5_I_F_xGoals_per60")
        == "mp_po_5on5_I_F_xGoals_per60"
    )


def test_enriched_forward_label_for_netfront_traits():
    name, summary = build_archetype_name_summary(
        0,
        [
            ("mp_reg_5on5_I_F_highDangerShots_per60", 1.2),
            ("mp_reg_5on5_I_F_reboundxGoals_per60", 1.1),
        ],
        [],
        group="forwards",
    )

    assert name == "Interior Net-Front Finisher"
    assert "high-danger" in summary
