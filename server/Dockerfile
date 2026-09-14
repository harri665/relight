FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install production dependencies only
RUN npm ci --only=production

# Copy source code
COPY . .

EXPOSE 3001

# Use node directly (no nodemon in production)
CMD ["node", "index.js"]