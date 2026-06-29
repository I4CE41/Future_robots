"""Wedge fingerprint detector for IR-Safety override.

A wedge is the state where the robot's front IR clearance is critical AND
both lateral IR sensors are saturated at their max range (1.0 m per
perception_fusion.py:50). The lateral tie-break in
perception_fusion.lateral_clearance becomes meaningless (both sides report
identical clearance), so the override's chosen turn direction is arbitrary
and the robot tends to scrape further into the corner instead of escaping.

The only safe action when this fingerprint is detected is to back out
immediately, regardless of the lateral tie-break.
"""


def is_wedged(perc, front_thresh=0.10, lateral_clip=0.95):
    """Return True iff the front is critical and both laterals are clipped.

    Args:
        perc: PerceptionResult from perception_fusion.PerceptionFusion.
        front_thresh: max front clearance (m) considered a wedge.
        lateral_clip: min lateral clearance (m) above which we treat the
            reading as saturated. Default 0.95 m, just below the 1.0 m
            ir_max_range defined in perception_fusion.py:50.

    Returns:
        bool: True if the wedge fingerprint matches.
    """
    return (perc.front_clearance < front_thresh
            and perc.left_clearance >= lateral_clip
            and perc.right_clearance >= lateral_clip)