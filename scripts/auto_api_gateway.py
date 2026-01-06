#!/usr/bin/env python3
"""
ΣLANG Phase 3: API Gateway Integration
======================================

Enterprise API management setup with load balancing and security.

Capabilities:
- Multi-gateway configuration
- Load balancing optimization
- Security policy automation
- Performance monitoring integration

Usage:
    python scripts/auto_api_gateway.py --kong --aws-api-gateway --azure-api-mgmt --load-balance
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import shutil

class APIGatewayIntegrator:
    """AI-powered API gateway integration system"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.gateway_dir = project_root / "api_gateways" / self.timestamp
        self.gateway_dir.mkdir(parents=True, exist_ok=True)

    def setup_gateways(self, gateways: List[str], load_balance: bool = True) -> Dict[str, Any]:
        """Setup API gateways for specified platforms"""
        print("🤖 SigmaLang Phase 3: API Gateway Integration")
        print("=" * 50)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Gateway Directory: {self.gateway_dir}")

        results = {}

        for gateway in gateways:
            print(f"\n[GATEWAY] Setting up {gateway.upper()}...")
            try:
                setup_result = self._setup_gateway(gateway, load_balance)
                results[gateway] = setup_result
                print(f"[SUCCESS] ✅ {gateway.upper()} gateway configured successfully")
            except Exception as e:
                print(f"[ERROR] ❌ {gateway.upper()} setup failed: {e}")
                results[gateway] = {"status": "failed", "error": str(e)}

        if load_balance and len(gateways) > 1:
            self._setup_load_balancing(gateways)

        return results

    def _setup_gateway(self, gateway: str, load_balance: bool) -> Dict[str, Any]:
        """Setup specific API gateway"""
        gateway_dir = self.gateway_dir / gateway
        gateway_dir.mkdir(exist_ok=True)

        if gateway == "kong":
            return self._setup_kong(gateway_dir)
        elif gateway == "aws-api-gateway":
            return self._setup_aws_api_gateway(gateway_dir)
        elif gateway == "azure-api-mgmt":
            return self._setup_azure_api_management(gateway_dir)
        else:
            raise ValueError(f"Unsupported gateway: {gateway}")

    def _setup_kong(self, gateway_dir: Path) -> Dict[str, Any]:
        """Setup Kong API Gateway"""
        # Kong declarative configuration
        kong_config = {
            "_format_version": "1.1",
            "services": [
                {
                    "name": "sigmalang-service",
                    "url": "http://sigmalang-backend:8000",
                    "routes": [
                        {
                            "name": "sigmalang-route",
                            "paths": ["/api/v1"],
                            "methods": ["GET", "POST", "PUT", "DELETE"]
                        }
                    ],
                    "plugins": [
                        {
                            "name": "cors",
                            "config": {
                                "origins": ["*"],
                                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                                "headers": ["Accept", "Accept-Version", "Content-Length", "Content-MD5", "Content-Type", "Date", "Authorization"],
                                "credentials": True
                            }
                        },
                        {
                            "name": "rate-limiting",
                            "config": {
                                "minute": 1000,
                                "hour": 50000
                            }
                        },
                        {
                            "name": "key-auth",
                            "config": {}
                        }
                    ]
                }
            ],
            "consumers": [
                {
                    "username": "sigmalang-user",
                    "keyauth_credentials": [
                        {
                            "key": "sigmalang-api-key"
                        }
                    ]
                }
            ]
        }

        config_file = gateway_dir / "kong.yml"
        with open(config_file, 'w') as f:
            json.dump(kong_config, f, indent=2)

        # Docker Compose for Kong
        docker_compose = {
            "version": "3.8",
            "services": {
                "kong": {
                    "image": "kong:latest",
                    "environment": {
                        "KONG_DATABASE": "off",
                        "KONG_DECLARATIVE_CONFIG": "/kong/declarative/kong.yml",
                        "KONG_PROXY_ACCESS_LOG": "/dev/stdout",
                        "KONG_ADMIN_ACCESS_LOG": "/dev/stdout",
                        "KONG_PROXY_ERROR_LOG": "/dev/stderr",
                        "KONG_ADMIN_ERROR_LOG": "/dev/stderr",
                        "KONG_ADMIN_LISTEN": "0.0.0.0:8001, 0.0.0.0:8444 ssl"
                    },
                    "ports": [
                        "8000:8000",
                        "8443:8443",
                        "8001:8001",
                        "8444:8444"
                    ],
                    "volumes": [
                        "./kong.yml:/kong/declarative/kong.yml"
                    ],
                    "healthcheck": {
                        "test": ["CMD", "kong", "health"],
                        "interval": "10s",
                        "timeout": "10s",
                        "retries": 10
                    }
                }
            }
        }

        compose_file = gateway_dir / "docker-compose.kong.yml"
        with open(compose_file, 'w') as f:
            json.dump(docker_compose, f, indent=2)

        return {
            "status": "success",
            "config": str(config_file),
            "compose": str(compose_file)
        }

    def _setup_aws_api_gateway(self, gateway_dir: Path) -> Dict[str, Any]:
        """Setup AWS API Gateway"""
        # CloudFormation template for API Gateway
        cf_template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Resources": {
                "SigmaLangApi": {
                    "Type": "AWS::ApiGateway::RestApi",
                    "Properties": {
                        "Name": "SigmaLangAPI",
                        "Description": "SigmaLang Semantic Compression API",
                        "EndpointConfiguration": {
                            "Types": ["REGIONAL"]
                        }
                    }
                },
                "CompressResource": {
                    "Type": "AWS::ApiGateway::Resource",
                    "Properties": {
                        "RestApiId": {"Ref": "SigmaLangApi"},
                        "ParentId": {"Fn::GetAtt": ["SigmaLangApi", "RootResourceId"]},
                        "PathPart": "compress"
                    }
                },
                "CompressMethod": {
                    "Type": "AWS::ApiGateway::Method",
                    "Properties": {
                        "RestApiId": {"Ref": "SigmaLangApi"},
                        "ResourceId": {"Ref": "CompressResource"},
                        "HttpMethod": "POST",
                        "AuthorizationType": "NONE",
                        "Integration": {
                            "Type": "HTTP",
                            "IntegrationHttpMethod": "POST",
                            "Uri": "http://sigmalang-backend:8000/compress",
                            "PassthroughBehavior": "WHEN_NO_TEMPLATES"
                        }
                    }
                },
                "ApiDeployment": {
                    "Type": "AWS::ApiGateway::Deployment",
                    "DependsOn": "CompressMethod",
                    "Properties": {
                        "RestApiId": {"Ref": "SigmaLangApi"},
                        "StageName": "prod"
                    }
                }
            },
            "Outputs": {
                "ApiUrl": {
                    "Description": "API Gateway URL",
                    "Value": {"Fn::Sub": "https://${SigmaLangApi}.execute-api.${AWS::Region}.amazonaws.com/prod"}
                }
            }
        }

        cf_file = gateway_dir / "api-gateway.yml"
        with open(cf_file, 'w') as f:
            json.dump(cf_template, f, indent=2)

        # API Gateway extensions for rate limiting
        extensions = {
            "swagger": "2.0",
            "info": {
                "title": "SigmaLang API",
                "version": "1.0.0"
            },
            "host": "api.sigmalang.com",
            "schemes": ["https"],
            "paths": {
                "/compress": {
                    "post": {
                        "summary": "Compress text",
                        "parameters": [
                            {
                                "name": "text",
                                "in": "body",
                                "required": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"}
                                    }
                                }
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Compressed result"
                            }
                        }
                    }
                }
            },
            "x-amazon-apigateway-request-validators": {
                "basic": {
                    "validateRequestBody": True,
                    "validateRequestParameters": True
                }
            },
            "x-amazon-apigateway-usage-plans": {
                "basic": {
                    "throttle": {
                        "burstLimit": 100,
                        "rateLimit": 50
                    },
                    "quota": {
                        "limit": 10000,
                        "offset": 0,
                        "period": "DAY"
                    }
                }
            }
        }

        swagger_file = gateway_dir / "api-gateway-extensions.json"
        with open(swagger_file, 'w') as f:
            json.dump(extensions, f, indent=2)

        return {
            "status": "success",
            "template": str(cf_file),
            "extensions": str(swagger_file)
        }

    def _setup_azure_api_management(self, gateway_dir: Path) -> Dict[str, Any]:
        """Setup Azure API Management"""
        # ARM template for APIM
        arm_template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "apimServiceName": {
                    "type": "string",
                    "metadata": {
                        "description": "Name of the API Management service"
                    }
                }
            },
            "resources": [
                {
                    "type": "Microsoft.ApiManagement/service",
                    "apiVersion": "2021-08-01",
                    "name": "[parameters('apimServiceName')]",
                    "location": "[resourceGroup().location]",
                    "sku": {
                        "name": "Developer",
                        "capacity": 1
                    },
                    "properties": {
                        "publisherEmail": "admin@sigmalang.com",
                        "publisherName": "SigmaLang Team"
                    }
                },
                {
                    "type": "Microsoft.ApiManagement/service/apis",
                    "apiVersion": "2021-08-01",
                    "name": "[concat(parameters('apimServiceName'), '/sigmalang-api')]",
                    "dependsOn": [
                        "[resourceId('Microsoft.ApiManagement/service', parameters('apimServiceName'))]"
                    ],
                    "properties": {
                        "displayName": "SigmaLang Semantic Compression API",
                        "apiRevision": "1",
                        "description": "Enterprise semantic compression service",
                        "subscriptionRequired": True,
                        "protocols": ["https"],
                        "path": "api/v1"
                    }
                }
            ]
        }

        arm_file = gateway_dir / "apim-template.json"
        with open(arm_file, 'w') as f:
            json.dump(arm_template, f, indent=2)

        # APIM policy configuration
        policy_xml = '''<policies>
    <inbound>
        <rate-limit calls="100" renewal-period="60" />
        <set-header name="X-API-Key" exists-action="override">
            <value>{{api-key}}</value>
        </set-header>
        <cors>
            <allowed-origins>
                <origin>*</origin>
            </allowed-origins>
            <allowed-methods>
                <method>GET</method>
                <method>POST</method>
                <method>PUT</method>
                <method>DELETE</method>
                <method>OPTIONS</method>
            </allowed-methods>
        </cors>
    </inbound>
    <backend>
        <forward-request />
    </backend>
    <outbound>
        <set-header name="X-Powered-By" exists-action="override">
            <value>SigmaLang API Gateway</value>
        </set-header>
    </outbound>
</policies>'''

        policy_file = gateway_dir / "apim-policy.xml"
        with open(policy_file, 'w') as f:
            f.write(policy_xml)

        return {
            "status": "success",
            "template": str(arm_file),
            "policy": str(policy_file)
        }

    def _setup_load_balancing(self, gateways: List[str]):
        """Setup load balancing across multiple gateways"""
        lb_config = {
            "version": "3.8",
            "services": {
                "load-balancer": {
                    "image": "nginx:alpine",
                    "ports": ["80:80", "443:443"],
                    "volumes": ["./nginx.conf:/etc/nginx/nginx.conf"],
                    "depends_on": gateways
                }
            }
        }

        # Generate nginx config for load balancing
        nginx_config = f'''events {{
    worker_connections 1024;
}}

http {{
    upstream sigmalang_backends {{
        {' '.join(f'server {gateway}:8000;' for gateway in gateways)}
    }}

    server {{
        listen 80;
        server_name api.sigmalang.com;

        location / {{
            proxy_pass http://sigmalang_backends;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
    }}
}}
'''

        lb_dir = self.gateway_dir / "load-balancer"
        lb_dir.mkdir(exist_ok=True)

        compose_file = lb_dir / "docker-compose.lb.yml"
        with open(compose_file, 'w') as f:
            json.dump(lb_config, f, indent=2)

        nginx_file = lb_dir / "nginx.conf"
        with open(nginx_file, 'w') as f:
            f.write(nginx_config)

        print(f"[LOAD-BALANCER] ✅ Load balancing configured for {len(gateways)} gateways")

def main():
    parser = argparse.ArgumentParser(description="SigmaLang API Gateway Integration")
    parser.add_argument("--kong", action="store_true", help="Setup Kong API Gateway")
    parser.add_argument("--aws-api-gateway", action="store_true", help="Setup AWS API Gateway")
    parser.add_argument("--azure-api-mgmt", action="store_true", help="Setup Azure API Management")
    parser.add_argument("--load-balance", action="store_true", default=True, help="Setup load balancing")

    args = parser.parse_args()

    # Determine gateways to setup
    gateways = []
    if args.kong:
        gateways.append("kong")
    if args.aws_api_gateway:
        gateways.append("aws-api-gateway")
    if args.azure_api_mgmt:
        gateways.append("azure-api-mgmt")

    if not gateways:
        gateways = ["kong", "aws-api-gateway", "azure-api-mgmt"]  # Default to all

    project_root = Path(__file__).parent.parent
    integrator = APIGatewayIntegrator(project_root)

    results = integrator.setup_gateways(gateways, args.load_balance)

    # Save results
    results_file = integrator.gateway_dir / "gateway_setup_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n🌐 API Gateway Integration Complete")
    print(f"📂 Results: {results_file}")

    # Summary
    successful = sum(1 for r in results.values() if r.get("status") == "success")
    total = len(results)
    print(f"✅ Configured {successful}/{total} API gateways successfully")

if __name__ == "__main__":
    main()