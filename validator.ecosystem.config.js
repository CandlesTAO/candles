module.exports = {
    apps: [{
      name: 'candles-validator',
      script: './start_validator.sh',
      watch: false,
      autorestart: true,
      max_restarts: 20,
      min_uptime: '30s',
      restart_delay: 2000,
      env: {
        NODE_ENV: 'production'
      },
      // Logging configuration
      log_file: './logs/validator.log',
      out_file: './logs/validator-out.log',
      error_file: './logs/validator-error.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      // Process management
      instances: 1,
      exec_mode: 'fork'
    }]
  }
