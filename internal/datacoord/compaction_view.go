package datacoord

import (
	"fmt"

	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/log"
)

type CompactionView interface {
	GetGroupLabel() *CompactionGroupLabel
	GetSegmentsView() []*SegmentView
	Append(segments ...*SegmentView)
	String() string
	Trigger() (CompactionView, string)
}

type FullViews struct {
	collections map[int64][]*SegmentView // collectionID
}

type SegmentViewSelector func(view *SegmentView) bool

func (v *FullViews) GetSegmentViewBy(collectionID UniqueID, selector SegmentViewSelector) []*SegmentView {
	views, ok := v.collections[collectionID]
	if !ok {
		return nil
	}

	var ret []*SegmentView

	for _, view := range views {
		if selector == nil || selector(view) {
			ret = append(ret, view.Clone())
		}
	}

	return ret
}

type CompactionGroupLabel struct {
	CollectionID UniqueID
	PartitionID  UniqueID
	Channel      string
}

func (label *CompactionGroupLabel) Key() string {
	return fmt.Sprintf("%d-%s", label.PartitionID, label.Channel)
}

func (label *CompactionGroupLabel) IsMinGroup() bool {
	return len(label.Channel) != 0 && label.PartitionID != 0 && label.CollectionID != 0
}

func (label *CompactionGroupLabel) Equal(other *CompactionGroupLabel) bool {
	return other != nil &&
		other.CollectionID == label.CollectionID &&
		other.PartitionID == label.PartitionID &&
		other.Channel == label.Channel
}

func (label *CompactionGroupLabel) String() string {
	return fmt.Sprintf("coll=%d, part=%d, channel=%s", label.CollectionID, label.PartitionID, label.Channel)
}

type SegmentView struct {
	ID UniqueID

	label *CompactionGroupLabel

	State commonpb.SegmentState
	Level datapb.SegmentLevel

	// positions
	startPos *msgpb.MsgPosition
	dmlPos   *msgpb.MsgPosition

	// size
	Size       float64
	ExpireSize float64
	DeltaSize  float64

	// file numbers
	BinlogCount   int
	StatslogCount int
	DeltalogCount int
}

func (s *SegmentView) Clone() *SegmentView {
	return &SegmentView{
		ID:            s.ID,
		label:         s.label,
		State:         s.State,
		Level:         s.Level,
		startPos:      s.startPos,
		dmlPos:        s.dmlPos,
		Size:          s.Size,
		ExpireSize:    s.ExpireSize,
		DeltaSize:     s.DeltaSize,
		BinlogCount:   s.BinlogCount,
		StatslogCount: s.StatslogCount,
		DeltalogCount: s.DeltalogCount,
	}
}

func GetViewsByInfo(segments ...*SegmentInfo) []*SegmentView {
	return lo.Map(segments, func(segment *SegmentInfo, _ int) *SegmentView {
		return &SegmentView{
			ID: segment.ID,
			label: &CompactionGroupLabel{
				CollectionID: segment.CollectionID,
				PartitionID:  segment.PartitionID,
				Channel:      segment.GetInsertChannel(),
			},

			State: segment.GetState(),
			Level: segment.GetLevel(),

			// positions
			startPos: segment.GetStartPosition(),
			dmlPos:   segment.GetDmlPosition(),

			DeltaSize:     GetBinlogSizeAsBytes(segment.GetDeltalogs()),
			DeltalogCount: GetBinlogCount(segment.GetDeltalogs()),

			Size:          GetBinlogSizeAsBytes(segment.GetBinlogs()),
			BinlogCount:   GetBinlogCount(segment.GetBinlogs()),
			StatslogCount: GetBinlogCount(segment.GetStatslogs()),

			// TODO: set the following
			// ExpireSize float64
		}
	})
}

func (v *SegmentView) Equal(other *SegmentView) bool {
	return v.Size == other.Size &&
		v.ExpireSize == other.ExpireSize &&
		v.DeltaSize == other.DeltaSize &&
		v.BinlogCount == other.BinlogCount &&
		v.StatslogCount == other.StatslogCount &&
		v.DeltalogCount == other.DeltalogCount
}

func (v *SegmentView) String() string {
	return fmt.Sprintf("ID=%d, label=<%s>, state=%s, level=%s, binlogSize=%.2f, binlogCount=%d, deltaSize=%.2f, deltaCount=%d, expireSize=%.2f",
		v.ID, v.label, v.State.String(), v.Level.String(), v.Size, v.BinlogCount, v.DeltaSize, v.DeltalogCount, v.ExpireSize)
}

func (v *SegmentView) LevelZeroString() string {
	return fmt.Sprintf("<ID=%d, level=%s, deltaSize=%.2f, deltaCount=%d>",
		v.ID, v.Level.String(), v.DeltaSize, v.DeltalogCount)
}

func GetBinlogCount(fieldBinlogs []*datapb.FieldBinlog) int {
	var num int
	for _, binlog := range fieldBinlogs {
		num += len(binlog.GetBinlogs())
	}
	return num
}

func GetExpiredSizeAsBytes(expireTime Timestamp, fieldBinlogs []*datapb.FieldBinlog) float64 {
	var expireSize float64
	for _, binlogs := range fieldBinlogs {
		for _, l := range binlogs.GetBinlogs() {
			// TODO, we should probably estimate expired log entries by total rows
			// in binlog and the ralationship of timeTo, timeFrom and expire time
			if l.TimestampTo < expireTime {
				log.Info("mark binlog as expired",
					zap.Int64("binlogID", l.GetLogID()),
					zap.Uint64("binlogTimestampTo", l.TimestampTo),
					zap.Uint64("compactExpireTime", expireTime))
				expireSize += float64(l.GetLogSize())
			}
		}
	}
	return expireSize
}

func GetBinlogSizeAsBytes(deltaBinlogs []*datapb.FieldBinlog) float64 {
	var deltaSize float64
	for _, deltaLogs := range deltaBinlogs {
		for _, l := range deltaLogs.GetBinlogs() {
			deltaSize += float64(l.GetLogSize())
		}
	}
	return deltaSize
}
