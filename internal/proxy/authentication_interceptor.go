package proxy

import (
	"context"
	"fmt"
	"strings"

	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util"
	"github.com/milvus-io/milvus/pkg/util/crypto"
	"github.com/milvus-io/milvus/pkg/util/merr"
)

func parseMD(rawToken string) (username, password string) {
	secrets := strings.SplitN(rawToken, util.CredentialSeperator, 2)
	if len(secrets) < 2 {
		log.Warn("invalid token format, length of secrets less than 2")
		return
	}
	username = secrets[0]
	password = secrets[1]
	return
}

func validSourceID(ctx context.Context, authorization []string) bool {
	if len(authorization) < 1 {
		// log.Warn("key not found in header", zap.String("key", util.HeaderSourceID))
		return false
	}
	// token format: base64<sourceID>
	token := authorization[0]
	sourceID, err := crypto.Base64Decode(token)
	if err != nil {
		return false
	}
	return sourceID == util.MemberCredID
}

// AuthenticationInterceptor verify based on kv pair <"authorization": "token"> in header
func AuthenticationInterceptor(ctx context.Context) (context.Context, error) {
	// The keys within metadata.MD are normalized to lowercase.
	// See: https://godoc.org/google.golang.org/grpc/metadata#New
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return nil, merr.WrapErrIoKeyNotFound("metadata", "auth check failure, due to occurs inner error: missing metadata")
	}
	if globalMetaCache == nil {
		return nil, merr.WrapErrServiceUnavailable("internal: Milvus Proxy is not ready yet. please wait")
	}
	// check:
	//	1. if rpc call from a member (like index/query/data component)
	// 	2. if rpc call from sdk
	if Params.CommonCfg.AuthorizationEnabled.GetAsBool() {
		if !validSourceID(ctx, md[strings.ToLower(util.HeaderSourceID)]) {
			authStrArr := md[strings.ToLower(util.HeaderAuthorize)]

			if len(authStrArr) < 1 {
				log.Warn("key not found in header")
				return nil, status.Error(codes.Unauthenticated, "missing authorization in header")
			}

			// token format: base64<username:password>
			// token := strings.TrimPrefix(authorization[0], "Bearer ")
			token := authStrArr[0]
			rawToken, err := crypto.Base64Decode(token)
			if err != nil {
				log.Warn("fail to decode the token", zap.Error(err))
				return nil, status.Error(codes.Unauthenticated, "invalid token format")
			}

			if !strings.Contains(rawToken, util.CredentialSeperator) {
				user, err := VerifyAPIKey(rawToken)
				if err != nil {
					log.Warn("fail to verify apikey", zap.Error(err))
					return nil, status.Error(codes.Unauthenticated, "auth check failure, please check api key is correct")
				}
				metrics.UserRPCCounter.WithLabelValues(user).Inc()
				userToken := fmt.Sprintf("%s%s%s", user, util.CredentialSeperator, "___")
				md[strings.ToLower(util.HeaderAuthorize)] = []string{crypto.Base64Encode(userToken)}
				ctx = metadata.NewIncomingContext(ctx, md)
			} else {
				// username+password authentication
				username, password := parseMD(rawToken)
				if !passwordVerify(ctx, username, password, globalMetaCache) {
					log.Warn("fail to verify password", zap.String("username", username))
					// NOTE: don't use the merr, because it will cause the wrong retry behavior in the sdk
					return nil, status.Error(codes.Unauthenticated, "auth check failure, please check username and password are correct")
				}
				metrics.UserRPCCounter.WithLabelValues(username).Inc()
			}
		}
	}
	return ctx, nil
}
