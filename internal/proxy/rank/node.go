package rank

//
//import (
//	"fmt"
//	"sync"
//
//	"go.uber.org/zap"
//
//	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
//	"github.com/milvus-io/milvus/pkg/log"
//	"github.com/milvus-io/milvus/pkg/util/timerecord"
//)
//
//type Node interface {
//	Operate(inputs map[UniqueID]*milvuspb.SearchResults) *milvuspb.SearchResults
//	Start()
//	Close()
//}
//
//type nodeCtx struct {
//	node Node
//
//	inputChannel map[UniqueID]chan *milvuspb.SearchResults
//
//	next    *nodeCtx
//	checker *timerecord.GroupChecker
//
//	closeCh chan struct{} // notify work to exit
//	closeWg sync.WaitGroup
//}
//
//func (c *nodeCtx) Start() {
//	c.closeWg.Add(1)
//	c.node.Start()
//	go c.work()
//}
//
//func (c *nodeCtx) Close() {
//	close(c.closeCh)
//	c.closeWg.Wait()
//}
//
//func (c *nodeCtx) work() {
//	defer c.closeWg.Done()
//	name := fmt.Sprintf("nodeCtxTtChecker-%s", c.node.Name())
//	if c.checker != nil {
//		c.checker.Check(name)
//		defer c.checker.Remove(name)
//	}
//
//	for {
//		select {
//		// close
//		case <-c.closeCh:
//			c.node.Close()
//			for _, ch := range c.inputChannel {
//				close(ch)
//			}
//			log.Debug("pipeline node closed", zap.String("nodeName", c.node.Name()))
//			return
//		case input := <-c.inputChannel:
//			var output Msg
//			output = c.node.Operate(input)
//			if c.checker != nil {
//				c.checker.Check(name)
//			}
//			if c.next != nil && output != nil {
//				c.next.inputChannel <- output
//			}
//		}
//	}
//}
//
//func newNodeCtx(node Node) *nodeCtx {
//	return &nodeCtx{
//		node:         node,
//		inputChannel: make(map[UniqueID]chan *milvuspb.SearchResults),
//		closeCh:      make(chan struct{}),
//		closeWg:      sync.WaitGroup{},
//	}
//}
//
//type Executor func(map[UniqueID]*milvuspb.SearchResults) *milvuspb.SearchResults
//
//type ExecNode struct {
//	collectionID int64
//	executor     Executor
//}
//
//func (node *ExecNode) Operate(inputs map[UniqueID]*milvuspb.SearchResults) *milvuspb.SearchResults {
//
//}
//
//func (node *ExecNode) Start() {
//
//}
//
//func (node *ExecNode) Close() {
//
//}
