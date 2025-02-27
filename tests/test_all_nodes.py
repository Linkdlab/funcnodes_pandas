from all_nodes_test_base import TestAllNodesBase


from test_df import (
    TestDataframeConvert,
    TestDataframeManipulation,
    TestDataframeMask,
    TestDataframeMath,
    TestDataFrameRowsCols,
    TestReduceDataFrameNode,
    TestDataframeOther,
)

from test_series import TestSeriesStrConvert, TestSeries
from test_grouping import TestGrouping


class TestAllNodes(TestAllNodesBase):
    sub_test_classes = [
        TestDataframeConvert,
        TestDataframeManipulation,
        TestDataframeMask,
        TestDataframeMath,
        TestDataFrameRowsCols,
        TestSeriesStrConvert,
        TestSeries,
        TestGrouping,
        TestReduceDataFrameNode,
        TestDataframeOther,
    ]
