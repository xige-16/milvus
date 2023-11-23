package datacoord

import (
	"math"
	"sync"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/pkg/log"
)

type IndexEngineVersionManager interface {
	Startup(sessions map[string]*sessionutil.Session)
	AddNode(session *sessionutil.Session)
	RemoveNode(session *sessionutil.Session)
	Update(session *sessionutil.Session)

	GetCurrentIndexEngineVersion() int32
	GetMinimalIndexEngineVersion() int32
}

type versionManagerImpl struct {
	mu       sync.Mutex
	versions map[int64]sessionutil.IndexEngineVersion
}

func newIndexEngineVersionManager() IndexEngineVersionManager {
	return &versionManagerImpl{
		versions: map[int64]sessionutil.IndexEngineVersion{},
	}
}

func (m *versionManagerImpl) Startup(sessions map[string]*sessionutil.Session) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, session := range sessions {
		m.addOrUpdate(session)
	}
}

func (m *versionManagerImpl) AddNode(session *sessionutil.Session) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.addOrUpdate(session)
}

func (m *versionManagerImpl) RemoveNode(session *sessionutil.Session) {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.versions, session.ServerID)
}

func (m *versionManagerImpl) Update(session *sessionutil.Session) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.addOrUpdate(session)
}

func (m *versionManagerImpl) addOrUpdate(session *sessionutil.Session) {
	log.Info("addOrUpdate version", zap.Int64("nodeId", session.ServerID), zap.Int32("minimal", session.IndexEngineVersion.MinimalIndexVersion), zap.Int32("current", session.IndexEngineVersion.CurrentIndexVersion))
	m.versions[session.ServerID] = session.IndexEngineVersion
}

func (m *versionManagerImpl) GetCurrentIndexEngineVersion() int32 {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.versions) == 0 {
		log.Info("index versions is empty")
		return 0
	}

	current := int32(math.MaxInt32)
	for _, version := range m.versions {
		if version.CurrentIndexVersion < current {
			current = version.CurrentIndexVersion
		}
	}
	log.Info("Merged current version", zap.Int32("current", current))
	return current
}

func (m *versionManagerImpl) GetMinimalIndexEngineVersion() int32 {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.versions) == 0 {
		log.Info("index versions is empty")
		return 0
	}

	minimal := int32(0)
	for _, version := range m.versions {
		if version.MinimalIndexVersion > minimal {
			minimal = version.MinimalIndexVersion
		}
	}
	log.Info("Merged minimal version", zap.Int32("minimal", minimal))
	return minimal
}
