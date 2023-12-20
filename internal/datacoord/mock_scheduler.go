// Code generated by mockery v2.32.4. DO NOT EDIT.

package datacoord

import mock "github.com/stretchr/testify/mock"

// MockScheduler is an autogenerated mock type for the Scheduler type
type MockScheduler struct {
	mock.Mock
}

type MockScheduler_Expecter struct {
	mock *mock.Mock
}

func (_m *MockScheduler) EXPECT() *MockScheduler_Expecter {
	return &MockScheduler_Expecter{mock: &_m.Mock}
}

// Finish provides a mock function with given fields: nodeID, planID
func (_m *MockScheduler) Finish(nodeID int64, planID int64) {
	_m.Called(nodeID, planID)
}

// MockScheduler_Finish_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Finish'
type MockScheduler_Finish_Call struct {
	*mock.Call
}

// Finish is a helper method to define mock.On call
//   - nodeID int64
//   - planID int64
func (_e *MockScheduler_Expecter) Finish(nodeID interface{}, planID interface{}) *MockScheduler_Finish_Call {
	return &MockScheduler_Finish_Call{Call: _e.mock.On("Finish", nodeID, planID)}
}

func (_c *MockScheduler_Finish_Call) Run(run func(nodeID int64, planID int64)) *MockScheduler_Finish_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64), args[1].(int64))
	})
	return _c
}

func (_c *MockScheduler_Finish_Call) Return() *MockScheduler_Finish_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockScheduler_Finish_Call) RunAndReturn(run func(int64, int64)) *MockScheduler_Finish_Call {
	_c.Call.Return(run)
	return _c
}

// GetTaskCount provides a mock function with given fields:
func (_m *MockScheduler) GetTaskCount() int {
	ret := _m.Called()

	var r0 int
	if rf, ok := ret.Get(0).(func() int); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(int)
	}

	return r0
}

// MockScheduler_GetTaskCount_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetTaskCount'
type MockScheduler_GetTaskCount_Call struct {
	*mock.Call
}

// GetTaskCount is a helper method to define mock.On call
func (_e *MockScheduler_Expecter) GetTaskCount() *MockScheduler_GetTaskCount_Call {
	return &MockScheduler_GetTaskCount_Call{Call: _e.mock.On("GetTaskCount")}
}

func (_c *MockScheduler_GetTaskCount_Call) Run(run func()) *MockScheduler_GetTaskCount_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockScheduler_GetTaskCount_Call) Return(_a0 int) *MockScheduler_GetTaskCount_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockScheduler_GetTaskCount_Call) RunAndReturn(run func() int) *MockScheduler_GetTaskCount_Call {
	_c.Call.Return(run)
	return _c
}

// LogStatus provides a mock function with given fields:
func (_m *MockScheduler) LogStatus() {
	_m.Called()
}

// MockScheduler_LogStatus_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'LogStatus'
type MockScheduler_LogStatus_Call struct {
	*mock.Call
}

// LogStatus is a helper method to define mock.On call
func (_e *MockScheduler_Expecter) LogStatus() *MockScheduler_LogStatus_Call {
	return &MockScheduler_LogStatus_Call{Call: _e.mock.On("LogStatus")}
}

func (_c *MockScheduler_LogStatus_Call) Run(run func()) *MockScheduler_LogStatus_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockScheduler_LogStatus_Call) Return() *MockScheduler_LogStatus_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockScheduler_LogStatus_Call) RunAndReturn(run func()) *MockScheduler_LogStatus_Call {
	_c.Call.Return(run)
	return _c
}

// Schedule provides a mock function with given fields:
func (_m *MockScheduler) Schedule() []*compactionTask {
	ret := _m.Called()

	var r0 []*compactionTask
	if rf, ok := ret.Get(0).(func() []*compactionTask); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*compactionTask)
		}
	}

	return r0
}

// MockScheduler_Schedule_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Schedule'
type MockScheduler_Schedule_Call struct {
	*mock.Call
}

// Schedule is a helper method to define mock.On call
func (_e *MockScheduler_Expecter) Schedule() *MockScheduler_Schedule_Call {
	return &MockScheduler_Schedule_Call{Call: _e.mock.On("Schedule")}
}

func (_c *MockScheduler_Schedule_Call) Run(run func()) *MockScheduler_Schedule_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockScheduler_Schedule_Call) Return(_a0 []*compactionTask) *MockScheduler_Schedule_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockScheduler_Schedule_Call) RunAndReturn(run func() []*compactionTask) *MockScheduler_Schedule_Call {
	_c.Call.Return(run)
	return _c
}

// Submit provides a mock function with given fields: t
func (_m *MockScheduler) Submit(t ...*compactionTask) {
	_va := make([]interface{}, len(t))
	for _i := range t {
		_va[_i] = t[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, _va...)
	_m.Called(_ca...)
}

// MockScheduler_Submit_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Submit'
type MockScheduler_Submit_Call struct {
	*mock.Call
}

// Submit is a helper method to define mock.On call
//   - t ...*compactionTask
func (_e *MockScheduler_Expecter) Submit(t ...interface{}) *MockScheduler_Submit_Call {
	return &MockScheduler_Submit_Call{Call: _e.mock.On("Submit",
		append([]interface{}{}, t...)...)}
}

func (_c *MockScheduler_Submit_Call) Run(run func(t ...*compactionTask)) *MockScheduler_Submit_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]*compactionTask, len(args)-0)
		for i, a := range args[0:] {
			if a != nil {
				variadicArgs[i] = a.(*compactionTask)
			}
		}
		run(variadicArgs...)
	})
	return _c
}

func (_c *MockScheduler_Submit_Call) Return() *MockScheduler_Submit_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockScheduler_Submit_Call) RunAndReturn(run func(...*compactionTask)) *MockScheduler_Submit_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockScheduler creates a new instance of MockScheduler. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockScheduler(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockScheduler {
	mock := &MockScheduler{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
