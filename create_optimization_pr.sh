Quando existe manage all:
	1. ALLOW GROUP Administrators to manage all-resources IN TENANCY where request.resource.tag != 'protected_api_resource:yes'
	2. ALLOW GROUP Administrators to manage vault IN TENANCY where request.networkSource.name='SODEXO_ALLOWED_IPs'
	3. ALLOW GROUP Administrators to manage buckets IN TENANCY where request.networkSource.name='SODEXO_ALLOWED_IPs'

Quando ja esta segregado:
	1. ALLOW GROUP Administrators to manage vault IN TENANCY where request.networkSource.name='SODEXO_ALLOWED_IPs'
	2. ALLOW GROUP Administrators to manage buckets IN TENANCY where request.networkSource.name='SODEXO_ALLOWED_IPs'

SODEXO_ALLOWED_IPs = [
	ForcePoint Proxy IP,
	Escritorio SP,
	OCI Outbound IP (via Jump),
	TCS IP????? / VCNs IP
]


REMOVE All v8 Policies


where request.networkSource.name='SODEXO_GLB_SIEM_QRADAR'