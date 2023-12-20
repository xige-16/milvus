// Code generated by mockery v2.32.4. DO NOT EDIT.

package mocks

import (
	context "context"

	grpc "google.golang.org/grpc"

	grpcclient "github.com/milvus-io/milvus/internal/util/grpcclient"

	mock "github.com/stretchr/testify/mock"

	sessionutil "github.com/milvus-io/milvus/internal/util/sessionutil"
)

// MockGrpcClient is an autogenerated mock type for the GrpcClient type
type MockGrpcClient[T grpcclient.GrpcComponent] struct {
	mock.Mock
}

type MockGrpcClient_Expecter[T grpcclient.GrpcComponent] struct {
	mock *mock.Mock
}

func (_m *MockGrpcClient[T]) EXPECT() *MockGrpcClient_Expecter[T] {
	return &MockGrpcClient_Expecter[T]{mock: &_m.Mock}
}

// Call provides a mock function with given fields: ctx, caller
func (_m *MockGrpcClient[T]) Call(ctx context.Context, caller func(T) (interface{}, error)) (interface{}, error) {
	ret := _m.Called(ctx, caller)

	var r0 interface{}
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, func(T) (interface{}, error)) (interface{}, error)); ok {
		return rf(ctx, caller)
	}
	if rf, ok := ret.Get(0).(func(context.Context, func(T) (interface{}, error)) interface{}); ok {
		r0 = rf(ctx, caller)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(interface{})
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, func(T) (interface{}, error)) error); ok {
		r1 = rf(ctx, caller)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockGrpcClient_Call_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Call'
type MockGrpcClient_Call_Call[T grpcclient.GrpcComponent] struct {
	*mock.Call
}

// Call is a helper method to define mock.On call
//   - ctx context.Context
//   - caller func(T)(interface{} , error)
func (_e *MockGrpcClient_Expecter[T]) Call(ctx interface{}, caller interface{}) *MockGrpcClient_Call_Call[T] {
	return &MockGrpcClient_Call_Call[T]{Call: _e.mock.On("Call", ctx, caller)}
}

func (_c *MockGrpcClient_Call_Call[T]) Run(run func(ctx context.Context, caller func(T) (interface{}, error))) *MockGrpcClient_Call_Call[T] {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(func(T) (interface{}, error)))
	})
	return _c
}

func (_c *MockGrpcClient_Call_Call[T]) Return(_a0 interface{}, _a1 error) *MockGrpcClient_Call_Call[T] {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockGrpcClient_Call_Call[T]) RunAndReturn(run func(context.Context, func(T) (interface{}, error)) (interface{}, error)) *MockGrpcClient_Call_Call[T] {
	_c.Call.Return(run)
	return _c
}

// Close provides a mock function with given fields:
func (_m *MockGrpcClient[T]) Close() error {
	ret := _m.Called()

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockGrpcClient_Close_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Close'
type MockGrpcClient_Close_Call[T grpcclient.GrpcComponent] struct {
	*mock.Call
}

// Close is a helper method to define mock.On call
func (_e *MockGrpcClient_Expecter[T]) Close() *MockGrpcClient_Close_Call[T] {
	return &MockGrpcClient_Close_Call[T]{Call: _e.mock.On("Close")}
}

func (_c *MockGrpcClient_Close_Call[T]) Run(run func()) *MockGrpcClient_Close_Call[T] {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockGrpcClient_Close_Call[T]) Return(_a0 error) *MockGrpcClient_Close_Call[T] {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockGrpcClient_Close_Call[T]) RunAndReturn(run func() error) *MockGrpcClient_Close_Call[T] {
	_c.Call.Return(run)
	return _c
}

// EnableEncryption provides a mock function with given fields:
func (_m *MockGrpcClient[T]) EnableEncryption() {
	_m.Called()
}

// MockGrpcClient_EnableEncryption_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'EnableEncryption'
type MockGrpcClient_EnableEncryption_Call[T grpcclient.GrpcComponent] struct {
	*mock.Call
}

// EnableEncryption is a helper method to define mock.On call
func (_e *MockGrpcClient_Expecter[T]) EnableEncryption() *MockGrpcClient_EnableEncryption_Call[T] {
	return &MockGrpcClient_EnableEncryption_Call[T]{Call: _e.mock.On("EnableEncryption")}
}

func (_c *MockGrpcClient_EnableEncryption_Call[T]) Run(run func()) *MockGrpcClient_EnableEncryption_Call[T] {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockGrpcClient_EnableEncryption_Call[T]) Return() *MockGrpcClient_EnableEncryption_Call[T] {
	_c.Call.Return()
	return _c
}

func (_c *MockGrpcClient_EnableEncryption_Call[T]) RunAndReturn(run func()) *MockGrpcClient_EnableEncryption_Call[T] {
	_c.Call.Return(run)
	return _c
}

// GetNodeID provides a mock function with given fields:
func (_m *MockGrpcClient[T]) GetNodeID() int64 {
	ret := _m.Called()

	var r0 int64
	if rf, ok := ret.Get(0).(func() int64); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(int64)
	}

	return r0
}

// MockGrpcClient_GetNodeID_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetNodeID'
type MockGrpcClient_GetNodeID_Call[T grpcclient.GrpcComponent] struct {
	*mock.Call
}

// GetNodeID is a helper method to define mock.On call
func (_e *MockGrpcClient_Expecter[T]) GetNodeID() *MockGrpcClient_GetNodeID_Call[T] {
	return &MockGrpcClient_GetNodeID_Call[T]{Call: _e.mock.On("GetNodeID")}
}

func (_c *MockGrpcClient_GetNodeID_Call[T]) Run(run func()) *MockGrpcClient_GetNodeID_Call[T] {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockGrpcClient_GetNodeID_Call[T]) Return(_a0 int64) *MockGrpcClient_GetNodeID_Call[T] {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockGrpcClient_GetNodeID_Call[T]) RunAndReturn(run func() int64) *MockGrpcClient_GetNodeID_Call[T] {
	_c.Call.Return(run)
	return _c
}

// GetRole provides a mock function with given fields:
func (_m *MockGrpcClient[T]) GetRole() string {
	ret := _m.Called()

	var r0 string
	if rf, ok := ret.Get(0).(func() string); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(string)
	}

	return r0
}

// MockGrpcClient_GetRole_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetRole'
type MockGrpcClient_GetRole_Call[T grpcclient.GrpcComponent] struct {
	*mock.Call
}

// GetRole is a helper method to define mock.On call
func (_e *MockGrpcClient_Expecter[T]) GetRole() *MockGrpcClient_GetRole_Call[T] {
	return &MockGrpcClient_GetRole_Call[T]{Call: _e.mock.On("GetRole")}
}

func (_c *MockGrpcClient_GetRole_Call[T]) Run(run func()) *MockGrpcClient_GetRole_Call[T] {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockGrpcClient_GetRole_Call[T]) Return(_a0 string) *MockGrpcClient_GetRole_Call[T] {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockGrpcClient_GetRole_Call[T]) RunAndReturn(run func() string) *MockGrpcClient_GetRole_Call[T] {
	_c.Call.Return(run)
	return _c
}

// ReCall provides a mock function with given fields: ctx, caller
func (_m *MockGrpcClient[T]) ReCall(ctx context.Context, caller func(T) (interface{}, error)) (interface{}, error) {
	ret := _m.Called(ctx, caller)

	var r0 interface{}
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, func(T) (interface{}, error)) (interface{}, error)); ok {
		return rf(ctx, caller)
	}
	if rf, ok := ret.Get(0).(func(context.Context, func(T) (interface{}, error)) interface{}); ok {
		r0 = rf(ctx, caller)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(interface{})
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, func(T) (interface{}, error)) error); ok {
		r1 = rf(ctx, caller)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockGrpcClient_ReCall_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'ReCall'
type MockGrpcClient_ReCall_Call[T grpcclient.GrpcComponent] struct {
	*mock.Call
}

// ReCall is a helper method to define mock.On call
//   - ctx context.Context
//   - caller func(T)(interface{} , error)
func (_e *MockGrpcClient_Expecter[T]) ReCall(ctx interface{}, caller interface{}) *MockGrpcClient_ReCall_Call[T] {
	return &MockGrpcClient_ReCall_Call[T]{Call: _e.mock.On("ReCall", ctx, caller)}
}

func (_c *MockGrpcClient_ReCall_Call[T]) Run(run func(ctx context.Context, caller func(T) (interface{}, error))) *MockGrpcClient_ReCall_Call[T] {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(func(T) (interface{}, error)))
	})
	return _c
}

func (_c *MockGrpcClient_ReCall_Call[T]) Return(_a0 interface{}, _a1 error) *MockGrpcClient_ReCall_Call[T] {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockGrpcClient_ReCall_Call[T]) RunAndReturn(run func(context.Context, func(T) (interface{}, error)) (interface{}, error)) *MockGrpcClient_ReCall_Call[T] {
	_c.Call.Return(run)
	return _c
}

// SetGetAddrFunc provides a mock function with given fields: _a0
func (_m *MockGrpcClient[T]) SetGetAddrFunc(_a0 func() (string, error)) {
	_m.Called(_a0)
}

// MockGrpcClient_SetGetAddrFunc_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetGetAddrFunc'
type MockGrpcClient_SetGetAddrFunc_Call[T grpcclient.GrpcComponent] struct {
	*mock.Call
}

// SetGetAddrFunc is a helper method to define mock.On call
//   - _a0 func()(string , error)
func (_e *MockGrpcClient_Expecter[T]) SetGetAddrFunc(_a0 interface{}) *MockGrpcClient_SetGetAddrFunc_Call[T] {
	return &MockGrpcClient_SetGetAddrFunc_Call[T]{Call: _e.mock.On("SetGetAddrFunc", _a0)}
}

func (_c *MockGrpcClient_SetGetAddrFunc_Call[T]) Run(run func(_a0 func() (string, error))) *MockGrpcClient_SetGetAddrFunc_Call[T] {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(func() (string, error)))
	})
	return _c
}

func (_c *MockGrpcClient_SetGetAddrFunc_Call[T]) Return() *MockGrpcClient_SetGetAddrFunc_Call[T] {
	_c.Call.Return()
	return _c
}

func (_c *MockGrpcClient_SetGetAddrFunc_Call[T]) RunAndReturn(run func(func() (string, error))) *MockGrpcClient_SetGetAddrFunc_Call[T] {
	_c.Call.Return(run)
	return _c
}

// SetNewGrpcClientFunc provides a mock function with given fields: _a0
func (_m *MockGrpcClient[T]) SetNewGrpcClientFunc(_a0 func(*grpc.ClientConn) T) {
	_m.Called(_a0)
}

// MockGrpcClient_SetNewGrpcClientFunc_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetNewGrpcClientFunc'
type MockGrpcClient_SetNewGrpcClientFunc_Call[T grpcclient.GrpcComponent] struct {
	*mock.Call
}

// SetNewGrpcClientFunc is a helper method to define mock.On call
//   - _a0 func(*grpc.ClientConn) T
func (_e *MockGrpcClient_Expecter[T]) SetNewGrpcClientFunc(_a0 interface{}) *MockGrpcClient_SetNewGrpcClientFunc_Call[T] {
	return &MockGrpcClient_SetNewGrpcClientFunc_Call[T]{Call: _e.mock.On("SetNewGrpcClientFunc", _a0)}
}

func (_c *MockGrpcClient_SetNewGrpcClientFunc_Call[T]) Run(run func(_a0 func(*grpc.ClientConn) T)) *MockGrpcClient_SetNewGrpcClientFunc_Call[T] {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(func(*grpc.ClientConn) T))
	})
	return _c
}

func (_c *MockGrpcClient_SetNewGrpcClientFunc_Call[T]) Return() *MockGrpcClient_SetNewGrpcClientFunc_Call[T] {
	_c.Call.Return()
	return _c
}

func (_c *MockGrpcClient_SetNewGrpcClientFunc_Call[T]) RunAndReturn(run func(func(*grpc.ClientConn) T)) *MockGrpcClient_SetNewGrpcClientFunc_Call[T] {
	_c.Call.Return(run)
	return _c
}

// SetNodeID provides a mock function with given fields: _a0
func (_m *MockGrpcClient[T]) SetNodeID(_a0 int64) {
	_m.Called(_a0)
}

// MockGrpcClient_SetNodeID_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetNodeID'
type MockGrpcClient_SetNodeID_Call[T grpcclient.GrpcComponent] struct {
	*mock.Call
}

// SetNodeID is a helper method to define mock.On call
//   - _a0 int64
func (_e *MockGrpcClient_Expecter[T]) SetNodeID(_a0 interface{}) *MockGrpcClient_SetNodeID_Call[T] {
	return &MockGrpcClient_SetNodeID_Call[T]{Call: _e.mock.On("SetNodeID", _a0)}
}

func (_c *MockGrpcClient_SetNodeID_Call[T]) Run(run func(_a0 int64)) *MockGrpcClient_SetNodeID_Call[T] {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64))
	})
	return _c
}

func (_c *MockGrpcClient_SetNodeID_Call[T]) Return() *MockGrpcClient_SetNodeID_Call[T] {
	_c.Call.Return()
	return _c
}

func (_c *MockGrpcClient_SetNodeID_Call[T]) RunAndReturn(run func(int64)) *MockGrpcClient_SetNodeID_Call[T] {
	_c.Call.Return(run)
	return _c
}

// SetRole provides a mock function with given fields: _a0
func (_m *MockGrpcClient[T]) SetRole(_a0 string) {
	_m.Called(_a0)
}

// MockGrpcClient_SetRole_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetRole'
type MockGrpcClient_SetRole_Call[T grpcclient.GrpcComponent] struct {
	*mock.Call
}

// SetRole is a helper method to define mock.On call
//   - _a0 string
func (_e *MockGrpcClient_Expecter[T]) SetRole(_a0 interface{}) *MockGrpcClient_SetRole_Call[T] {
	return &MockGrpcClient_SetRole_Call[T]{Call: _e.mock.On("SetRole", _a0)}
}

func (_c *MockGrpcClient_SetRole_Call[T]) Run(run func(_a0 string)) *MockGrpcClient_SetRole_Call[T] {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockGrpcClient_SetRole_Call[T]) Return() *MockGrpcClient_SetRole_Call[T] {
	_c.Call.Return()
	return _c
}

func (_c *MockGrpcClient_SetRole_Call[T]) RunAndReturn(run func(string)) *MockGrpcClient_SetRole_Call[T] {
	_c.Call.Return(run)
	return _c
}

// SetSession provides a mock function with given fields: sess
func (_m *MockGrpcClient[T]) SetSession(sess *sessionutil.Session) {
	_m.Called(sess)
}

// MockGrpcClient_SetSession_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetSession'
type MockGrpcClient_SetSession_Call[T grpcclient.GrpcComponent] struct {
	*mock.Call
}

// SetSession is a helper method to define mock.On call
//   - sess *sessionutil.Session
func (_e *MockGrpcClient_Expecter[T]) SetSession(sess interface{}) *MockGrpcClient_SetSession_Call[T] {
	return &MockGrpcClient_SetSession_Call[T]{Call: _e.mock.On("SetSession", sess)}
}

func (_c *MockGrpcClient_SetSession_Call[T]) Run(run func(sess *sessionutil.Session)) *MockGrpcClient_SetSession_Call[T] {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*sessionutil.Session))
	})
	return _c
}

func (_c *MockGrpcClient_SetSession_Call[T]) Return() *MockGrpcClient_SetSession_Call[T] {
	_c.Call.Return()
	return _c
}

func (_c *MockGrpcClient_SetSession_Call[T]) RunAndReturn(run func(*sessionutil.Session)) *MockGrpcClient_SetSession_Call[T] {
	_c.Call.Return(run)
	return _c
}

// NewMockGrpcClient creates a new instance of MockGrpcClient. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockGrpcClient[T grpcclient.GrpcComponent](t interface {
	mock.TestingT
	Cleanup(func())
}) *MockGrpcClient[T] {
	mock := &MockGrpcClient[T]{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
