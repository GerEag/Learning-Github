public class WeatherInfo{
	double temp;
	double humidity;
	public WeatherInfo(double temp, double humidity) {
		super();
		this.temp = temp;
		this.humidity = humidity;
	}
	public double getTemp() {
		return temp;
	}
	public void setTemp(double temp) {
		this.temp = temp;
	}
	public double getHumidity() {
		return humidity;
	}
	public void setHumidity(double humidity) {
		this.humidity = humidity;
	}
	
	
}