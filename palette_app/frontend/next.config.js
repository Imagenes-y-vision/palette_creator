/** @type {import('next').NextConfig} */
const nextConfig = {
    images: {
	domains: ['localhost'],
        remotePatterns: [
          {
            protocol: 'https',
            hostname: 'images.pexels.com',
            port: '',
            pathname: '**',
          },
        ],
      },
}

module.exports = nextConfig
