pipeline {
    agent any
    
    stages {
        stage('CLoning from Github') {
            steps {
                script {
                    echo 'CLoning from Github...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/data-guru0/FINAL-COURSE-.git']])
                }
            }
        }
    }
}