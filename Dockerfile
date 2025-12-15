# Use official lightweight Node.js image
FROM node:18-alpine

# Create app directory
WORKDIR /app

# Install app dependencies (uses package-lock.json if present)
COPY package*.json ./
RUN npm ci --only=production

# Copy app source
COPY . .

# Expose port (adjust if your app uses a different one)
EXPOSE 3000

# Start the app (expects a "start" script in package.json)
CMD ["npm", "start"]